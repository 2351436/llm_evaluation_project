from transformers import AutoTokenizer, AutoModel
import torch
import time
import os

model_name_or_path = "/mnt/data/chatglm3-6b"

# --- 最终测试用 Prompt (共10个问题) ---
prompts_to_test = {
    "问题1 (冬天夏天)": "请说出以下两句话区别在哪里? 1、冬天:能穿多少穿多少 2、夏天:能穿多少穿多少",
    "问题2 (单身狗)": "请说出以下两句话区别在哪里?单身狗产生的原因有两个,一是谁都看不上,二是谁都看不上",
    "问题3 (谁不知道)": "他知道我知道你知道他不知道吗? 这句话里,到底谁不知道",
    "问题4 (明明白白)": "明明明明明白白白喜欢他,可她就是不说。 这句话里,明明和白白谁喜欢谁?",
    "问题5 (意思)": "领导:你这是什么意思? 小明:没什么意思。意思意思。 领导:你这就不够意思了。 小明:小意思,小意思。领导:你这人真有意思。 小明:其实也没有别的意思。 领导:那我就不好意思了。 小明:是我不好意思。请问:以上\"意思\"分别是什么意思。",
    "问题6 (Python偶数求和)": "我想用Python写一个函数，它可以接收一个数字列表，然后返回这个列表里所有偶数的和。你能帮我写出这个函数吗？",
    "问题7 (光合作用解释)": "什么是“光合作用”？请用小学生也能听懂的语言简单解释一下它的过程和意义。",
    "问题8 (果园数学题)": "一个果园里有苹果树120棵，梨树比苹果树少25棵，桃树的数量是梨树的2倍。请问桃树有多少棵？请列出计算步骤。",
    "问题9 (AI文本摘要)": "请概括以下这段文字的主要内容，不超过60个字：“人工智能（AI）正在迅速改变世界。从自动驾驶汽车到医疗诊断，AI的应用无处不在，深刻影响着经济结构和社会生活。然而，随着AI能力的增强，关于其伦理影响、就业冲击、数据隐私和潜在安全风险的讨论也日益激烈。确保AI技术的负责任发展和公平应用，是当前面临的重要挑战。”",
    "问题10 (城市导游推荐)": "假设你是一位城市导游，请向一位首次来访的外国游客推荐三个你所在城市（或任选一个中国知名城市）最值得游览的景点，并简要说明推荐理由。"
}

print(f"DSW 实例当前 Python 环境中的 torch 版本: {torch.__version__}")
print(f"PyTorch 是否使用 CUDA: {torch.cuda.is_available()}")

# 尝试导入并应用 IPEX 优化
ipex_available = False
try:
    import intel_extension_for_pytorch as ipex
    ipex_available = True
    print(f"Intel Extension for PyTorch (IPEX) 版本: {ipex.__version__} 已成功导入。")
except ImportError:
    print("警告: Intel Extension for PyTorch (IPEX) 未安装或无法导入。模型将在标准 PyTorch CPU 上运行，性能可能较低。")
    print("如果希望获得更好的 CPU 推理性能, 请尝试安装 IPEX: pip install intel-extension-for-pytorch")

print(f"准备从以下路径加载 Tokenizer 和模型: {model_name_or_path}")
print("这可能需要一些时间，请耐心等待...")
load_start_time = time.time()

try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    print("Tokenizer 加载成功。")
except Exception as e:
    print(f"加载 Tokenizer 失败: {e}")
    print(f"请检查路径 '{model_name_or_path}' 是否正确，以及模型文件是否完整。")
    exit()

try:
    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, # ChatGLM3 官方推荐 bfloat16 以获得更好性能和减少显存占用，CPU上也适用
        device_map="cpu"
    ).eval()
    print("模型加载成功 (使用 bfloat16)。")
except Exception as load_err:
    print(f"使用 bfloat16 加载模型失败: {load_err}")
    print("尝试使用 float32 加载模型...")
    try:
        model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu"
        ).eval()
        print("模型加载成功 (使用 float32)。")
    except Exception as e:
        print(f"使用 float32 加载模型也失败: {e}")
        print(f"请检查路径 '{model_name_or_path}' 是否正确、文件是否完整。")
        print("也请确保 PyTorch CPU 版本已正确安装，并且有足够的内存。")
        exit()

# 应用 IPEX 优化 (如果可用)
if ipex_available:
    try:
        print("正在尝试应用 IPEX 优化...")
        model = ipex.optimize(model, dtype=model.config.torch_dtype if hasattr(model.config, "torch_dtype") else torch.bfloat16)
        print("IPEX 优化成功应用。")
    except Exception as e:
        print(f"IPEX 优化失败: {e}")
        print("模型将不使用 IPEX 优化运行。")

load_end_time = time.time()
print(f"模型和 Tokenizer 加载耗时: {load_end_time - load_start_time:.2f} 秒")
print("-" * 30)

total_questions_start_time = time.time()
for i, (name, current_prompt) in enumerate(prompts_to_test.items()):
    print(f"\n--- 测试问题 {i+1}/{len(prompts_to_test)}: {name} ---")
    print(f"Prompt: {current_prompt}")
    print("模型回答:")
    
    question_start_time = time.time()
    # ChatGLM3 的 stream_chat 方法更适合流式对话
    current_history = []
    try:
        # 使用 model.stream_chat 进行流式交互
        # stream_chat 返回一个生成器
        full_response = ""
        for response_chunk, history in model.stream_chat(tokenizer, current_prompt, history=current_history,
                                                        max_length=2048, # ChatGLM3 的 max_length 指的是上下文总长度
                                                        do_sample=True, temperature=0.7, top_p=0.8):
            print(response_chunk.replace(full_response, ""), end="", flush=True) # 打印增量部分
            full_response = response_chunk
        current_history = history # 更新历史记录，尽管在这个单轮测试脚本中可能不是必须的
        print() # 确保换行
    except Exception as e:
        print(f"\n模型生成过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
    question_end_time = time.time()
    print(f"\n(问题 \"{name}\" 回答完毕，耗时: {question_end_time - question_start_time:.2f} 秒)")
    print("=" * 50 + "\n")

total_questions_end_time = time.time()
print(f"所有 {len(prompts_to_test)} 个 ChatGLM3-6B 问题测试完成。总耗时: {(total_questions_end_time - total_questions_start_time)/60:.2f} 分钟。")
