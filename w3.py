import time, numpy as np, wave
import speech_engine as se
w = wave.open('test_data/speech/vi_150s.wav')
audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)/32768.0
eng = se.SpeechEngine(model_size=se.LIVE_MODEL, window_seconds=2.0)
eng._model = se._whisper_model(se.LIVE_MODEL)
t0=time.time(); first=None; windows=0
for start in range(0, 40*16000, 16000):
    blk = audio[start:start+16000].copy()
    eng._buffer.append((blk, len(blk))); eng._buffered += len(blk)
    while True:
        got = eng._take(eng.window_seconds)
        if got is None: break
        u = eng.buffer_usage()
        eng._process_window(got); windows += 1
        if first is None and not eng._final.empty():
            first = time.time()-t0
            print(f'FIRST sentence at {first:.1f}s of wall time', flush=True)
eng._flush_pending()
out=[]
while not eng._final.empty(): out.append(eng._final.get())
print(f'40s audio -> {windows} windows, {len(out)} sentences, {time.time()-t0:.1f}s wall', flush=True)
for s in out[:4]: print('   ', s[:66], flush=True)
