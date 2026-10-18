//! One embedding process per user, shared by every `engram` and `engram-cli` process.
//!
//! Each process that embeds would otherwise load its own copy of the model (about 50 MB
//! resident), so concurrent sessions, and every hook invocation, pay for it again. The
//! first process that needs an embedding starts `<its own binary> __embed-serve`, which
//! loads the model once, answers over a Unix socket in the user's runtime directory, and
//! exits after `ENGRAM_EMBED_IDLE_SECS` (default 900) without a request.
//!
//! Protocol: one JSON object per line. The client sends `{"texts": [...]}` with texts
//! already prefixed; the server answers `{"vectors": [[...], ...]}` or `{"error": "..."}`.
//! The socket name carries the crate version, so a process from another release (which
//! may embed differently) never talks to this one's server.

#[cfg(not(unix))]
use crate::error::MemoryError;

/// The argument that makes `engram` or `engram-cli` run the embedding server instead of
/// its normal entry point.
pub const SERVE_ARG: &str = "__embed-serve";

/// When the current process was started as the embedding server, run it and exit. Called
/// first thing in both binaries' `main`.
pub fn run_if_requested() {
    if std::env::args().nth(1).as_deref() == Some(SERVE_ARG) {
        #[cfg(unix)]
        let code = match unix::serve() {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("[engram] embedding server: {e}");
                1
            }
        };
        #[cfg(not(unix))]
        let code = 2;
        std::process::exit(code);
    }
}

#[cfg(unix)]
pub use unix::Client;

/// Without Unix sockets there is no shared process; [`Client::from_env`] declines and
/// every process embeds in-process.
#[cfg(not(unix))]
pub struct Client;

#[cfg(not(unix))]
impl Client {
    pub fn from_env() -> Option<Self> {
        None
    }

    pub fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, MemoryError> {
        unreachable!("no shared embedding process without Unix sockets")
    }
}

#[cfg(unix)]
mod unix {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::os::unix::process::CommandExt;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use serde::{Deserialize, Serialize};

    use super::SERVE_ARG;
    use crate::embedding::Model;
    use crate::error::MemoryError;

    const DEFAULT_IDLE_SECS: u64 = 900;

    /// How long a client waits for a server it just started. The first start downloads
    /// the model, about 23 MB; after that, loading takes under a second.
    const STARTUP_WAIT: Duration = Duration::from_secs(30);

    /// A request that takes longer than this is treated as a hung server.
    const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

    #[derive(Serialize, Deserialize)]
    struct Request {
        texts: Vec<String>,
    }

    #[derive(Serialize, Deserialize)]
    struct Response {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        vectors: Option<Vec<Vec<f32>>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    }

    fn io_error(context: &str, e: impl std::fmt::Display) -> MemoryError {
        MemoryError::Embedding(format!("{context}: {e}"))
    }

    /// `$XDG_RUNTIME_DIR/engram`, else `~/.cache/engram`; created owner-only, since
    /// anyone who can reach the socket can make the server compute.
    fn runtime_dir() -> Result<PathBuf, MemoryError> {
        let base = std::env::var_os("XDG_RUNTIME_DIR")
            .map(PathBuf::from)
            .or_else(dirs::cache_dir)
            .ok_or_else(|| MemoryError::Embedding("no runtime or cache directory".into()))?;
        let dir = base.join("engram");
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&dir)
            .map_err(|e| io_error("creating the runtime directory", e))?;
        Ok(dir)
    }

    fn socket_name() -> String {
        format!("embed-{}", env!("CARGO_PKG_VERSION"))
    }

    /// `sun_path` holds 108 bytes on Linux and 104 on macOS, including the terminator.
    const MAX_SOCKET_PATH: usize = 100;

    pub struct Client {
        socket: PathBuf,
        /// Set once the shared process has failed this process: by then the model is
        /// loaded in-process, and retrying the socket would only add latency to every call.
        failed: AtomicBool,
        /// Serializes starts within this process.
        starting: Mutex<()>,
    }

    impl Client {
        /// `None` when `ENGRAM_EMBED_SHARED` is `0`/`false`/`no` or there is nowhere to put
        /// the socket.
        pub fn from_env() -> Option<Self> {
            if matches!(
                std::env::var("ENGRAM_EMBED_SHARED").as_deref(),
                Ok("0") | Ok("false") | Ok("no")
            ) {
                return None;
            }
            let socket = runtime_dir().ok()?.join(format!("{}.sock", socket_name()));
            if socket.as_os_str().len() > MAX_SOCKET_PATH {
                return None;
            }
            Some(Self {
                socket,
                failed: AtomicBool::new(false),
                starting: Mutex::new(()),
            })
        }

        pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, MemoryError> {
            if self.failed.load(Ordering::Relaxed) {
                return Err(MemoryError::Embedding(
                    "shared embedding process failed earlier".into(),
                ));
            }
            let result = match UnixStream::connect(&self.socket) {
                Ok(stream) => Ok(stream),
                Err(_) => self.start_server(),
            }
            .and_then(|stream| request(stream, texts));
            if result.is_err() {
                self.failed.store(true, Ordering::Relaxed);
            }
            result
        }

        /// Start the server and wait for its socket; it may have exited idle since this
        /// process last used it. Gives up as soon as the server exits with an error rather
        /// than waiting out the deadline; one that exits cleanly lost the start race to
        /// another, whose socket is coming.
        fn start_server(&self) -> Result<UnixStream, MemoryError> {
            let _starting = self
                .starting
                .lock()
                .map_err(|e| MemoryError::Embedding(format!("start lock poisoned: {e}")))?;
            // Another thread of this process may have started it while this one waited.
            if let Ok(stream) = UnixStream::connect(&self.socket) {
                return Ok(stream);
            }
            let mut child = spawn()?;
            let deadline = Instant::now() + STARTUP_WAIT;
            let mut exited_cleanly = false;
            loop {
                if let Ok(stream) = UnixStream::connect(&self.socket) {
                    reap(child);
                    return Ok(stream);
                }
                if !exited_cleanly {
                    match child.try_wait() {
                        Ok(Some(status)) if status.success() => exited_cleanly = true,
                        Ok(Some(status)) => {
                            return Err(MemoryError::Embedding(format!(
                                "embedding server exited with {status}; see {}",
                                self.socket.with_extension("log").display()
                            )));
                        }
                        _ => {}
                    }
                }
                if Instant::now() >= deadline {
                    reap(child);
                    return Err(MemoryError::Embedding(
                        "embedding server did not start in time".into(),
                    ));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }

    /// Wait for the server in the background, so it never lingers as a zombie of a
    /// long-lived client.
    fn reap(mut child: std::process::Child) {
        std::thread::spawn(move || child.wait());
    }

    fn request(stream: UnixStream, texts: &[String]) -> Result<Vec<Vec<f32>>, MemoryError> {
        stream
            .set_read_timeout(Some(REQUEST_TIMEOUT))
            .map_err(|e| io_error("socket", e))?;
        let mut line = serde_json::to_string(&Request {
            texts: texts.to_vec(),
        })?;
        line.push('\n');
        (&stream)
            .write_all(line.as_bytes())
            .map_err(|e| io_error("sending to the embedding server", e))?;
        let mut reply = String::new();
        BufReader::new(&stream)
            .read_line(&mut reply)
            .map_err(|e| io_error("reading from the embedding server", e))?;
        let response: Response = serde_json::from_str(&reply)?;
        match (response.vectors, response.error) {
            (Some(vectors), _) if vectors.len() == texts.len() => Ok(vectors),
            (_, Some(error)) => Err(MemoryError::Embedding(error)),
            _ => Err(MemoryError::Embedding(
                "embedding server returned the wrong number of vectors".into(),
            )),
        }
    }

    /// Start `<this binary> __embed-serve` in its own session, so a client killed by its
    /// caller (a hook timeout) does not take the server with it.
    fn spawn() -> Result<std::process::Child, MemoryError> {
        let exe = std::env::current_exe().map_err(|e| io_error("locating this binary", e))?;
        let log = runtime_dir()?.join(format!("{}.log", socket_name()));
        let stderr = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log)
            .map(Stdio::from)
            .unwrap_or_else(|_| Stdio::null());
        let mut command = Command::new(exe);
        command
            .arg(SERVE_ARG)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(stderr);
        // SAFETY: `setsid` is async-signal-safe and touches no memory of the parent; it is
        // the only call made between fork and exec.
        unsafe {
            command.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
        command
            .spawn()
            .map_err(|e| io_error("starting the embedding server", e))
    }

    /// Hold `<socket>.lock` for the life of the server: two clients may both start one,
    /// and the loser exits instead of replacing the winner's socket.
    fn take_lock(dir: &Path) -> Result<Option<std::fs::File>, MemoryError> {
        use std::os::fd::AsRawFd;
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(dir.join(format!("{}.lock", socket_name())))
            .map_err(|e| io_error("opening the lock file", e))?;
        // SAFETY: `flock` on a descriptor this function owns, kept open by returning it.
        let locked = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0;
        Ok(locked.then_some(file))
    }

    pub(super) fn serve() -> Result<(), MemoryError> {
        let dir = runtime_dir()?;
        let Some(_lock) = take_lock(&dir)? else {
            return Ok(());
        };
        let socket = dir.join(format!("{}.sock", socket_name()));
        let model = Arc::new(Mutex::new(Model::load()?));

        // A socket file left by a server that died; the lock proves nobody owns it.
        let _ = std::fs::remove_file(&socket);
        let listener = UnixListener::bind(&socket).map_err(|e| io_error("binding", e))?;
        std::fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600))
            .map_err(|e| io_error("socket permissions", e))?;

        let idle = Duration::from_secs(
            std::env::var("ENGRAM_EMBED_IDLE_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_IDLE_SECS),
        );
        let started = Instant::now();
        let last_request = Arc::new(AtomicU64::new(0));
        {
            let last_request = Arc::clone(&last_request);
            let socket = socket.clone();
            std::thread::spawn(move || {
                loop {
                    std::thread::sleep(Duration::from_secs(5).min(idle));
                    let since = started.elapsed()
                        - Duration::from_millis(last_request.load(Ordering::Relaxed));
                    if since >= idle {
                        let _ = std::fs::remove_file(&socket);
                        std::process::exit(0);
                    }
                }
            });
        }

        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let model = Arc::clone(&model);
            let last_request = Arc::clone(&last_request);
            std::thread::spawn(move || {
                let mut reader = BufReader::new(&stream);
                let mut line = String::new();
                if reader.read_line(&mut line).is_err() {
                    return;
                }
                last_request.store(started.elapsed().as_millis() as u64, Ordering::Relaxed);
                let response = match serde_json::from_str::<Request>(&line) {
                    Ok(req) => match model.lock() {
                        Ok(mut model) => match model.embed(&req.texts) {
                            Ok(vectors) => Response {
                                vectors: Some(vectors),
                                error: None,
                            },
                            Err(e) => Response {
                                vectors: None,
                                error: Some(e.to_string()),
                            },
                        },
                        Err(e) => Response {
                            vectors: None,
                            error: Some(format!("model lock poisoned: {e}")),
                        },
                    },
                    Err(e) => Response {
                        vectors: None,
                        error: Some(format!("bad request: {e}")),
                    },
                };
                last_request.store(started.elapsed().as_millis() as u64, Ordering::Relaxed);
                if let Ok(mut reply) = serde_json::to_string(&response) {
                    reply.push('\n');
                    let _ = (&stream).write_all(reply.as_bytes());
                }
            });
        }
        Ok(())
    }
}
