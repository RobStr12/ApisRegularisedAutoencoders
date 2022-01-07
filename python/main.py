import os
import time

start_global = time.time()
os.system("project_setup.py")
os.system("data_cleanup.py")

for tr in ["SP", "TH"]:
    os.system(f"python ./Model_training.py -p {tr} -d cuda")

os.system("python Encoding.py")

end_global = time.time()

run_time = end_global - start_global

print(f"Total runtime: {int(run_time // 60)}:{run_time % 60}")
