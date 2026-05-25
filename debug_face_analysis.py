import time
print("Importing FaceAnalysis...")
start = time.time()
from insightface.app import FaceAnalysis
print(f"Imported in {time.time() - start:.2f}s")

start = time.time()
print("Instantiating FaceAnalysis with allowed_modules=['detection', 'recognition']...")
app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
print(f"Instantiated in {time.time() - start:.2f}s")

start = time.time()
print("Preparing FaceAnalysis app...")
app.prepare(ctx_id=0, det_size=(640, 640))
print(f"Prepared in {time.time() - start:.2f}s")

print("Done! Models in app:", app.models.keys())
