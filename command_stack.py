import subprocess
import time
from itertools import product
from multiprocessing import Semaphore, Process

def run_command(iid, pid, bchm_id, semaphore):
    command = f"nohup python discovery.py {iid} {pid} {bchm_id} &"
    try:
        print(f"Starting: {command}")
        subprocess.run(command, shell=True, check=True)
        print(f"Completed: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Failed: {command} with error: {e}")
    finally:
        semaphore.release()

def main():
    max_processes = 12
    semaphore = Semaphore(max_processes)
    
    iids = range(2, 6)  # 2到5
    pids = [1, 3, 4, 5, 16, 23]
    bchm_ids = range(0, 14)  # 0到13
    
    processes = []
    
    for iid, pid, bchm_id in product(iids, pids, bchm_ids):
        semaphore.acquire()
        p = Process(target=run_command, args=(iid, pid, bchm_id, semaphore))
        p.start()
        processes.append(p)
        # 添加短暂延迟以避免资源竞争
        time.sleep(1)
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
