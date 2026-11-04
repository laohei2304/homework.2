import torch
import torch.nn.functional as F
import soundfile as sf
import argparse
from models import SincNetTransformer

def predict(wav_path, weight_path="weights/best_sincnet_transformer.pkl"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    class_names = checkpoint['class_names']
    model = SincNetTransformer(num_machines=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    data, sr = sf.read(wav_path, dtype='float32', always_2d=True)
    waveform = torch.from_numpy(data.T).mean(dim=0, keepdim=True)
    n_samples = 16000 * 4
    if waveform.size(1) < n_samples: waveform = F.pad(waveform, (0, n_samples - waveform.size(1)))
    else: waveform = waveform[:, :n_samples]
    waveform = waveform.unsqueeze(0).to(device)
    with torch.no_grad():
        m_logits, a_logits = model(waveform)
        m_probs = F.softmax(m_logits, dim=1)
        a_probs = F.softmax(a_logits, dim=1)
        pred_machine = class_names[m_probs.argmax().item()]
        is_abnormal = a_probs[0, 1].item() > 0.5
        print("-" * 40)
        print(f"预测机器类型: {pred_machine} (置信度: {m_probs.max().item():.4f})")
        print(f"是否异常: {is_abnormal} (异常概率: {a_probs[0, 1].item():.4f})")
        print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_path", type=str, required=True)
    args = parser.parse_args()
    predict(args.wav_path)