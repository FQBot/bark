from IPython.display import Audio
import nltk  # we'll use this to split into sentences
import numpy as np

from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE

preload_models()
script = """
杭州非秦科技有限公司成立于2019年,专注于人工智能和自然语言处理技术。
我们的核心技术包括:
1. 智能对话交互应用(SophiaPal for Chat):实现各行业的自动化语音交互,提高服务效率。适用于客户服务、商业交易、医疗咨询、教育培训等行业。
2. 智能助理应用(SophiaPal for Bot):为各企业和个人提供智能助手服务,自动处理常见任务,提高工作效率。适用于企业办公、客户服务、商业运营、个人生活助手等场景。
3. 智能创作应用(SophiaPal for Write):作为人类作者的智能写作伙伴,提高内容创作效率和质量。适用于新闻写作、论文写作、内容营销、文艺创作、广告创意等领域。
4. 虚拟数字人应用(SophiaPal for vHuman):通过高度融合虚实世界,延伸人类生命体验。形象丰富多变,满足从人到机的生命延续。适用于虚拟主播、虚拟偶像、虚拟员工、智能教学、社交娱乐和医疗康复等场景。
我们致力于让人工智能走入“真实世界”,改变生产和生活方式。虚拟数字人技术是实现这个愿景的重要手段,需要多学科技术协同推进。
""".replace("\n", " ").strip()
sentences = nltk.sent_tokenize(script)
SPEAKER = "v2/zh_speaker_0"
silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

pieces = []
for sentence in sentences:
    audio_array = generate_audio(sentence, history_prompt=SPEAKER)
    pieces += [audio_array, silence.copy()]



# save audio to disk
audio_pieces = []
silence_pieces = []
for sentence in sentences:
    audio_array = generate_audio(sentence, history_prompt=SPEAKER)
    audio_pieces.append(audio_array.reshape(-1, 1))
    silence_pieces.append(silence.reshape(-1, 1))  # 添加reshape(-1, 1)

audio_pieces = np.concatenate(audio_pieces, axis=1)
silence_pieces = np.concatenate(silence_pieces, axis=1)

pieces = np.concatenate((audio_pieces, silence_pieces), axis=0)
write_wav("bark_generation.wav", SAMPLE_RATE, pieces)

Audio(np.concatenate(pieces), rate=SAMPLE_RATE)
