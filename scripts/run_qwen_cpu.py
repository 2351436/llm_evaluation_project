from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
import torch
import time

# 新增: 尝试导入 IPEX
ipex_available = False
try:
    import intel_extension_for_pytorch as ipex
    ipex_available = True
    print(f"Intel Extension for PyTorch (IPEX) 版本: {ipex.__version__} 已成功导入。")
except ImportError:
    print("警告: Intel Extension for PyTorch (IPEX) 未安装或无法导入。模型将在标准 PyTorch CPU 上运行，性能可能较低。")
    print("如果希望获得更好的 CPU 推理性能, 请尝试安装 IPEX: pip install intel-extension-for-pytorch")

model_name_or_path = "/mnt/data/Qwen-7B-Chat"

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

print(f"准备从以下路径加载 Tokenizer 和模型: {model_name_or_path}")
print("这可能需要一些时间，请耐心等待...")
load_start_time = time.time()
tokenizer = None
model = None

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
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype="auto", # IPEX 优化时会参考这个
        device_map="cpu"    # 确保模型加载到 CPU
    ).eval()
    print("模型加载成功。")
except Exception as e:
    print(f"加载模型失败: {e}")
    print(f"请检查路径 '{model_name_or_path}' 是否正确、文件是否完整。")
    print("也请确保 PyTorch CPU 版本已正确安装，并且有足够的内存。")
    exit()

load_end_time = time.time()
print(f"模型和 Tokenizer 加载耗时: {load_end_time - load_start_time:.2f} 秒")
print("-" * 30)

# 应用 IPEX 优化 (如果可用)
if ipex_available and model: # 确保 model 已成功加载
    try:
        print("正在尝试应用 IPEX 优化...")
        # Qwen 通常使用 bfloat16 或 float16。对于 CPU IPEX，bfloat16 通常更好。
        # 我们从模型配置中获取原始 dtype，如果 IPEX 支持，它会尝试使用。
        original_dtype = model.config.torch_dtype if hasattr(model.config, "torch_dtype") else torch.bfloat16
        print(f"模型原始 dtype: {original_dtype}")
        
        if next(model.parameters()).device.type == 'cpu':
            # 对于 AutoModelForCausalLM，推荐的优化方法是 ipex.optimize(model, dtype=dtype, inplace=True)
            # 或者 model = ipex.optimize(model, dtype=dtype)
            # 确保使用正确的 dtype，bfloat16 通常是 CPU 上的好选择
            model = ipex.optimize(model, dtype=torch.bfloat16) # 强制使用 bfloat16，如果原始是 float32，IPEX 会处理转换
            print(f"IPEX 优化成功应用 (尝试使用 dtype: torch.bfloat16)。")
        else:
            print(f"IPEX 优化跳过：模型不在 CPU 上 (device: {next(model.parameters()).device})。请确保 device_map='cpu'。")

    except Exception as e:
        print(f"IPEX 优化失败: {e}")
        import traceback
        traceback.print_exc()
        print("模型将不使用 IPEX 优化运行。")
else:
    if not ipex_available:
        print("IPEX 未导入，模型将不使用 IPEX 优化运行。")
    if not model:
        print("模型未加载，跳过 IPEX 优化。")


print("-" * 30)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

total_questions_start_time = time.time()
for i, (name, current_prompt) in enumerate(prompts_to_test.items()):
    print(f"\n--- 测试问题 {i+1}/{len(prompts_to_test)}: {name} ---")
    print(f"Prompt: {current_prompt}")
    print("模型回答 (流式输出):")
    
    question_start_time = time.time()
    # 确保 inputs 在 CPU 上
    inputs = tokenizer(current_prompt, return_tensors="pt").input_ids.to("cpu")

    generation_kwargs = {
        "input_ids": inputs,
        "streamer": streamer,
        "max_new_tokens": 768, 
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
    }
    
    if tokenizer.pad_token_id is None: 
        # Qwen 的 tokenizer 可能没有 pad_token_id，这时通常用 eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id 
    
    generation_kwargs["pad_token_id"] = tokenizer.pad_token_id # 显式传递

    try:
        # 对于 IPEX 优化过的模型，直接调用 generate
        with torch.no_grad(): # 在推理时使用 no_grad 可以节省内存并可能加速
            model.generate(**generation_kwargs)
    except Exception as e:
        print(f"\n模型生成过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    question_end_time = time.time()
    print(f"\n(问题 \"{name}\" 回答完毕，耗时: {question_end_time - question_start_time:.2f} 秒)")
    print("=" * 50 + "\n")

total_questions_end_time = time.time()
print(f"所有 {len(prompts_to_test)} 个 Qwen-7B-Chat 问题测试完成。总耗时: {(total_questions_end_time - total_questions_start_time)/60:.2f} 分钟。")

