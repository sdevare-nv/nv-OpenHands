import libtmux
import psutil
import threading
import time
import os
import sys
import uuid


class TmuxMemoryMonitor(threading.Thread):
    def __init__(self, tmux_server, limit_mb, interval=0.5):
        super().__init__(daemon=True)
        self.server = tmux_server
        self.limit_mb = limit_mb
        self.limit_bytes = limit_mb * 1024 * 1024
        self.interval = interval
        self.running = True
        self.kill_triggered = False

    def _get_server_pid(self):
        try:
            pid_str = self.server.cmd("display-message", "-p", "#{pid}").stdout[0]
            return int(pid_str)
        except:
            return None

    def get_tree_memory(self, parent_pid):
        total_mem = 0
        try:
            parent = psutil.Process(parent_pid)
            # recursive=True finds children AND grandchildren
            procs = [parent] + parent.children(recursive=True)
            for p in procs:
                try:
                    total_mem += p.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except psutil.NoSuchProcess:
            return 0
        return total_mem

    def kill_inner_processes(self):
        try:
            for session in self.server.sessions:
                for window in session.windows:
                    for pane in window.panes:
                        try:
                            # 1. Get Shell PID inside the pane
                            pane_pid = int(
                                pane.cmd("display-message", "-p", "#{pane_pid}").stdout[
                                    0
                                ]
                            )
                            parent = psutil.Process(pane_pid)

                            # 2. Get EVERY descendant (Main script + Background procs)
                            children = parent.children(recursive=True)

                            print(
                                f"[GUARD] Found {len(children)} descendants in Pane {pane.id} (Shell PID: {pane_pid})"
                            )

                            # 3. Log and Kill them all
                            for child in children:
                                try:
                                    # Grab info before killing
                                    pid = child.pid
                                    cmd = " ".join(child.cmdline())

                                    print(
                                        f"[GUARD] 🗡️  Killing PID: {pid:<6} CMD: {cmd}..."
                                    )  # Truncate long cmds
                                    child.kill()
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    print(f"[GUARD] ⚠️  Process vanished before kill.")

                        except (psutil.NoSuchProcess, IndexError, ValueError):
                            continue
        except Exception as e:
            print(f"[GUARD] Error: {e}")

    def run(self):
        print(f"[GUARD] Monitoring started. Limit: {self.limit_mb}MB")
        time.sleep(1)
        server_pid = self._get_server_pid()
        if not server_pid:
            return

        while self.running:
            try:
                used_bytes = self.get_tree_memory(server_pid)
            except:
                break

            used_mb = used_bytes / (1024 * 1024)
            print(f"[GUARD] Tree Usage: {int(used_mb)}MB / {self.limit_mb}MB")

            if used_bytes > self.limit_bytes:
                print(
                    f"\n[GUARD] 🚨 LIMIT EXCEEDED ({int(used_mb)}MB)! Killing inner processes..."
                )
                self.kill_inner_processes()
                self.kill_triggered = True
                time.sleep(5)

            time.sleep(self.interval)


# --- 2. LEAKER (With Background Procs) ---
def create_leaker_script(filename="leaker_verbose.py"):
    script_content = """
import multiprocessing
import subprocess
import time
import os
import sys

def consume_memory_worker(mb):
    try:
        data = bytearray(mb * 1024 * 1024)
        while True: time.sleep(1)
    except: pass

if __name__ == "__main__":
    processes = []
    print(f"--- Main Process PID: {os.getpid()} ---")

    # 1. Spawn a sneaky BACKGROUND process
    print("--- Spawning Background Process (100MB) ---")
    bg_code = "import time; a = bytearray(100*1024*1024); time.sleep(1000)"
    # We use -c to run inline python code
    bg_proc = subprocess.Popen([sys.executable, "-c", bg_code])
    processes.append(bg_proc)

    # 2. Spawn 8 Standard Workers (50MB each)
    print("--- Spawning 8 Standard Workers ---")
    for i in range(100):
        p = multiprocessing.Process(target=consume_memory_worker, args=(50,))
        p.start()
        processes.append(p)
        time.sleep(0.2)

    print("--- ALL STARTED. Waiting to be killed... ---")
    try:
        while True: time.sleep(1)
    except: pass
"""
    with open(filename, "w") as f:
        f.write(script_content)
    return os.path.abspath(filename)


# --- 3. TEST RUNNER ---
def run_test():
    leaker_file = create_leaker_script()
    unique_socket = f"/tmp/tmux-test-{uuid.uuid4()}"

    print(f"1. Starting Tmux Server (Socket: {unique_socket})")
    server = libtmux.Server(socket_path=unique_socket)

    session = server.new_session(session_name="verbose_test", start_directory="/tmp")
    pane = session.active_pane
    server_pid = int(server.cmd("display-message", "-p", "#{pid}").stdout[0])

    print("2. Sending Leaker command...")
    pane.send_keys(f"{sys.executable} {leaker_file}")

    # Limit 300MB. Leaker uses ~500MB
    print("3. Starting Guard (Limit: 300MB)")
    guard = TmuxMemoryMonitor(server, limit_mb=1000, interval=0.5)
    guard.start()

    start_time = time.time()
    try:
        while time.time() - start_time < 30:
            if guard.kill_triggered:
                print("\n[TEST] Kill triggered. Verifying cleanup...")
                time.sleep(3)

                if not psutil.pid_exists(server_pid):
                    print("❌ FAIL: Tmux Server died!")
                    return

                current_mem_mb = guard.get_tree_memory(server_pid) / (1024 * 1024)
                print(f"[TEST] Residual Memory: {int(current_mem_mb)}MB")

                if current_mem_mb < 50:
                    print(
                        "✅ PASS: Memory dropped. All processes (including background) killed."
                    )
                    return
                else:
                    print("❌ FAIL: Memory still high!")
                    return

            time.sleep(1)
        print("\n❌ FAIL: Timeout.")

    finally:
        if os.path.exists(leaker_file):
            os.remove(leaker_file)
        guard.running = False
        try:
            server.kill_server()
        except:
            pass


if __name__ == "__main__":
    run_test()
