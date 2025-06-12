import threading
import queue
import asyncio
from functools import partial


def custom_print(current_queue, *args, **kwargs):
    output = " ".join(str(arg) for arg in args)
    current_queue.put(output)


class CodeExecutor:
    def __init__(self, executor_id):
        self.executor_id = executor_id
        self.code_queue = queue.Queue()
        self.async_output_queue = queue.Queue()
        self.code_ready = threading.Event()
        self.execution_context = {}
        self.inactivity_timer = None
        self.inactivity_limit = 600
        self.thread = threading.Thread(target=self.code_execution_thread, daemon=True)
        self.thread.start()
        self.status = "running"
        self.latest_global_variable = globals().copy()
        self.lock = threading.Lock()  # Lock for thread safety

    def reset_inactivity_timer(self):
        if self.inactivity_timer:
            self.inactivity_timer.cancel()
        self.inactivity_timer = threading.Timer(
            self.inactivity_limit, self.stop_executor
        )
        self.inactivity_timer.start()

    def stop_executor(self):
        self.submit_code("exit", self.latest_global_variable)
    def code_execution_thread(self):
        self.reset_inactivity_timer()
        while True:
            self.code_ready.wait()
            self.code_ready.clear()
            with self.lock:
                code_to_execute, current_global_var = self.code_queue.get()
            for key in current_global_var:
                self.execution_context[key] = current_global_var[key]
            if code_to_execute == "exit":
                print(f"Current Executor ({self.executor_id}) executor stopping.")
                if self.inactivity_timer:
                    self.inactivity_timer.cancel()
                self.status = "stopped"
                break
            try:
                exec_code = compile(code_to_execute, '<string>', 'exec')
                if asyncio.iscoroutinefunction(exec_code):
                    asyncio.run(exec(exec_code, self.execution_context))
                else:
                    exec(exec_code, self.execution_context)
            except Exception as e:
                print(f"Error executing code: {e}")
            finally:
                self.async_output_queue.put(None)
            self.code_queue.task_done()
            self.reset_inactivity_timer()

    async def async_output(self):
        while True:
            try:
                output = self.async_output_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.1)
                continue
            if output is None:
                break
            yield output

    def submit_code(self, code, global_variable):
        self.latest_global_variable = global_variable
        global_variable["print"] = partial(custom_print, self.async_output_queue)
        with self.lock:
            self.code_queue.put((code, global_variable))
        self.code_ready.set()
        self.reset_inactivity_timer()


class ExecutorManager:
    def __init__(self):
        self.executors = {}
        self.lock = threading.Lock()  # Add a lock for thread-safe access

    def get_or_create_executor(self, executor_id):
        with self.lock:  # Ensure thread-safe access to the executors dictionary
            newly_created_executor = False
            if executor_id not in self.executors:
                newly_created_executor = True
                self.executors[executor_id] = CodeExecutor(executor_id)
            else:
                if self.executors[executor_id].status == "stopped":
                    del self.executors[executor_id]
                    self.executors[executor_id] = CodeExecutor(executor_id)
                    newly_created_executor = True
            print(
                f"Executor {executor_id} created. Newly created: {newly_created_executor}"
            )
            return newly_created_executor, self.executors[executor_id]

    def clean_up(self, executor_id):
        with self.lock:  # Ensure thread-safe access to the executors dictionary
            if executor_id in self.executors:
                self.executors[executor_id].submit_code("exit")
                del self.executors[executor_id]
            else:
                print(f"Executor {executor_id} not found.")
