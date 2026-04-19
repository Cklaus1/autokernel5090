import shutil, os
UTILS = "/tmp/flash-attention/hopper/utils.h"
BACKUP = UTILS + ".bak"
if os.path.exists(BACKUP):
    shutil.copy(BACKUP, UTILS)
    print("Reverted utils.h")
else:
    print("No backup to revert")
