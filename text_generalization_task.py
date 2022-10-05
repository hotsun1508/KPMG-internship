# %% [markdown]
# # [Task #1] 텍스트 데이터 일반화 

# %% [markdown]
# > ## 텍스트 데이터 불러오기
# - Dart 0.1 데이터 사용

# %%
print("================== START =================")
import datetime
now = datetime.datetime.now()
print("\t", now)

# %% [markdown]
# - GPU 사용

# %%
import torch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is ready for you! ")

if use_cuda:  
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
    print('__Device Information:',torch.cuda.device(0))

# %% [markdown]
# #### 1. dart_train.txt, dart_valid.txt 불러오기

# %% [markdown]
# - 데이터 10,000개씩만 읽어오기 

# %%
train_text = []
valid_text = []
lines = 10000 if use_cuda == True else 260  # gpu로 돌릴 땐 1만개 사용
# lines = 10000  # gpu로 돌릴 땐 1만개 사용
print(lines)

with open('dart_train.txt','r', encoding='UTF8') as file:
    for line in range(lines): 
        train_text.append(file.readline())

with open('dart_valid.txt','r', encoding='UTF8') as file:
    for line in range(lines): 
        valid_text.append(file.readline())

# %% [markdown]
# > ## 정규표현식 사용하여 데이터 전환하기 
# - label data 생성
# - [STS Normalizaton](notion://www.notion.so/STS-Normalization-3b33a9f34bcc45338c6d2180fbb55933) 기준

# %% [markdown]
# #### 1. 정규표현식 함수화하기 

# %%
# import 라이브러리 
import re

# %% [markdown]
# - 금액

# %%
# monetary unit -> 999원 
def unit1(x):
    """
    ₩000,000,000 원
    ₩ 72,800,000원
    ¥ 125,000,000 원
    $198,000,000원
    $ 55,000,000원
    W7,920.000 원정
    W79,200,000
    """
    pattern = r'[(w|W|₩|$|¥)]\s*((\d+),)+\d+(.\d+)\s*(원정*|원*)'
    replacement = "999원"

    return re.sub(pattern, replacement, x)

# 한국어 표현 A
def unit2(x):
    """
    4천5백만원
    3억 3천 2백만원
    4천5백만원
    2조2천억원
    99억원
    130억달러
    10조 7,999원
    436억불
    1,699,567백만원
    4,170억원
    """
    pattern = r'(\d+,?[조억천만백십]{0,2}\s{0,1})?(\d+,?)?(\d+[조억천만백십]{0,2}\s{0,1})([원달러불]{1,2})'
    replacement = "999원"

    return re.sub(pattern, replacement, x)

# 한국어 표현 B
def unit3(x):
    """
    일억이천오백만원정
    칠천이백팔십만원 정
    구억일천육백만원
    오천오백만 원정
    오천오백만 원
    오천오백만원
    천오백육십이만 원정
    천사백이십만 원정
    백사십이만 원정
    """
    pattern = r'[일이삼사오육칠팔구십백천만억조]{2,}\s{0,1}원(\s{0,1}정){0,1}'
    replacement = "999원"

    return re.sub(pattern, replacement, x)

# %% [markdown]
# - 일자

# %%
# 년, 월,일 → 2022년, 7월, 20일
def year(x):
    """
    2918년
    18년
    2022 년
    12 년
    """
    pattern = r'\d{2,4}\s{0,1}년'
    replacement = "2022년"

    return re.sub(pattern, replacement, x)

def month(x):
    """
    1월
    12월
    2 월
    12 월
    """
    pattern = r'\d{1,2}\s{0,1}월'
    replacement = "7월"

    return re.sub(pattern, replacement, x)

def day(x):
    """
    12일
    2 일
    12 일
    123 일
    """
    pattern = r'\d{1,3}\s{0,1}일'
    replacement = "20일"

    return re.sub(pattern, replacement, x)

# Y.M.D, Y/M/D, Y-M-D → 2022년 7월 20일
def date(x):
    """
    2017. 4.10
    2017-4-10
    2017/ 4/10
    2013-08-08
    """
    pattern = r'\d{4}\s{0,1}(\.|-|/)\s{0,1}\d{1,2}\s{0,1}(\.|-|/)\s{0,1}\d{1,2}'
    replacement = "2022년 7월 20일"

    return re.sub(pattern, replacement, x)

# ~개월, 주 → 3개월
def months(x):
    """
    21개월, 21 개월, 23 주
    """
    pattern = r'\d{1,2}\s{0,1}(개월|주)'
    replacement = "3개월" 

    return re.sub(pattern, replacement, x)

# 10 영업일 → 7 영업일
def days(x):
    """
    10 영업일
    """
    pattern = r'\d{1,5}\s{0,1}(영업일)'
    replacement = "7 영업일"

    return re.sub(pattern, replacement, x)


# %% [markdown]
# - 순서

# %%
# 가. i. I.  → 가.
def ordinal1(x):
    """
    I. 냐냐냔
    III.후후후후
    IV.이히히히 
    i. 우리는 밥을 먹는다
    ii. 우리는 뭐를 한다.
    가. 우리는 무엇을 한다.
    """
    pattern = r'^([가-핳]|i{1,10}|V{0,1}I{1,3}V{0,3})\.'
    replacement = "가."

    return re.sub(pattern, replacement, x)

# 가) i) I) → 가.
def ordinal2(x):
    """
    I) 냐냐냔
    III)후후후후
    IV)이히히히 
    i) 우리는 밥을 먹는다
    ii) 우리는 뭐를 한다.
    가) 우리는 무엇을 한다.
    """
    pattern = r'^([가-핳]|i{1,10}|V{0,1}I{1,3}V{0,3})\)'
    replacement = "가."

    return re.sub(pattern, replacement, x)

# (가), (i), (I) → 가.
def ordinal3(x):
    """
    (III)후후후후
    (IV)이히히히 
    (i) 우리는 밥을 먹는다
    (ii) 우리는 뭐를 한다.
    (가) 우리는 무엇을 한다.
    """
    pattern = r'^\(([가-핳]|i{1,10}|V{0,1}I{1,3}V{0,3})\)'
    replacement = "가."

    return re.sub(pattern, replacement, x)

# %% [markdown]
# - 숫자

# %%
### 분수 : 0/0, 0분의 05 → 2분의 1
def fraction(x):
    """
    1/2
    123 / 132
    123 분의 2
    2분의5
    2.1분의 2
    99.22 / 123.23
    """
    pattern = r'\d{1,9}((.)\d{0,9}){0,1}\s{0,1}(\/|분의)\s{0,1}\d{1,9}((.)\d{0,9}){0,1}'
    replacement = "2분의 1"

    return re.sub(pattern, replacement, x)


### 퍼센트 -> 10%
def percentage(x):
    """
    108%
    108 %
    삼십%
    삼십 퍼센트
    이십퍼센트
    21 퍼센트
    """
    pattern = r'(((\d+),)*\d*(.\d*)|[일이삼사오육칠팔구십백천만억조]+)\s{0,1}(%|퍼센트)'
    replacement = "10%"

    return re.sub(pattern, replacement, x)

# %% [markdown]
# #### 2. 정규표현식 함수들을 리스트 하나에 담기

# %%
regex_list = [unit1, unit2, unit3, year, month, day, date, months, days, ordinal1, ordinal2, ordinal3, fraction, percentage]
print("================== regex list =================")

# %% [markdown]
# #### 3. Pipeline 만들기 
# - 모든 정규식 한번에 실행 가능한 함수 

# %%
# x = 입력받은 문장 

def pipeline(x, regex_list):
    for regex in regex_list:
        x = regex(x)
    return x 

# %% [markdown]
# #### 4. 변환된 label 텍스트 모두 리스트에 저장 
# - train_label_text, valid_label_text 
print("================== Regex pipeline =================")

# %%
train_label_text = [pipeline(sentence, regex_list) for sentence in train_text]  # 공백 제거 
valid_label_text = [pipeline(sentence, regex_list) for sentence in valid_text]  # 공백 제거 

print("================== Regex func application =================")

# %% [markdown]
# #### 5. 텍스트 파일로 저장하기
# - dart_train_label.txt, dart_valid_label.txt

# %%
# 쓰기모드 w, writelines(리스트)
with open('dart_train_label.txt', 'w', encoding='UTF8') as f:
    for line in train_label_text:
        f.write(line)

with open('dart_valid_label.txt', 'w', encoding='UTF8') as f:
    for line in valid_label_text:
        f.write(line)
print("================== Saved label text =================")

# %%
import joblib

print(joblib.dump(train_label_text, 'joblib_train_label.pkl'))
print(joblib.dump(valid_label_text, 'joblib_valid_label.pkl'))

# %% [markdown]
# > ## 데이터 전처리

# %% [markdown]
# #### 1. Input값을 Dictionary 형태로 만들기

# %%
def to_dict(inputs, targets):
    dic = []
    for i, t in zip(inputs, targets):
        item = {
            "inputs": i,
            "targets": t
        }
        dic.append(item)
    
    return dic

# %%
train_dic = to_dict(train_text, train_label_text)
valid_dic = to_dict(valid_text, valid_label_text)

# %% [markdown]
# #### 2. 허깅페이스의 [kobart mini](https://huggingface.co/cosmoquester/bart-ko-mini) pretrained model 사용

# %%
model_checkpoint = "./kobart_mini"  
# model_checkpoint = "cosmoquester/bart-ko-mini"  # w/ internet connection

# %%
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,  cache_dir="download")

# %%
max_input_length = 128
max_target_length = 128

def preprocess_function(sentences):
    inputs = [each["inputs"] for each in sentences]
    targets = [each["targets"] for each in sentences]
    
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding=True)

    # target 데이터 토큰화
    labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding=True)

    # label 데이터를 model inputs 딕셔너리에 추가 
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# %% [markdown]
# #### 3. Train 데이터와 valid 데이터 나누기
# - `input_ids`, `attention_mask`, `labels`

# %%
# Train data 
train_in = preprocess_function(train_dic)

# Valid data
valid_in = preprocess_function(valid_dic)

# %% [markdown]
# #### 4. 입력데이터와 정답 데이터 나누기
# - `train_input`, `train_label`, `valid_input`, `valid_label` 

# %%
def split_xy(data):
    input = {
        "input_ids" : data['input_ids'],
        "attention_mask" : data['attention_mask']
    }

    label = data['labels']
    return input, label

# %%
# Train data 
train_input, train_label = split_xy(train_in)

# Valid data
valid_input, valid_label = split_xy(valid_in)

# %% [markdown]
# #### 5. PyTorch Dataset 객체 생성하는 함수 사용

# %%
import torch
class Dart_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_feats, data_labels, input_keys):
        self.feats = data_feats
        self.labels = data_labels
        self.input_keys = input_keys

    def __getitem__(self, idx):
        item = {key: torch.tensor(self.feats[key][idx]) for key in self.input_keys}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# %%
print("================== Pytorch Dataset =================")

# Train data
train_dataset = Dart_Dataset(train_input, train_label, train_input.keys())

# Valid data
val_dataset = Dart_Dataset(valid_input, valid_label, valid_input.keys())

# %% [markdown]
# - 잘 나오는지 확인

# %%
''.join(tokenizer.convert_ids_to_tokens(train_dataset[5]['input_ids']))

# %%
print("================== Save train dataset, val dataset =================")

print(joblib.dump(list(train_dataset), 'train_dataset.pkl'))
print(joblib.dump(list(val_dataset), 'val_dataset.pkl'))

# %% [markdown]
# > ## 모델 학습 설정
# - 모델 inference 단계

# %% [markdown]
# #### 1. 평가지표 

# %% [markdown]
# #### Bleu로 돌려보기 

# %%
from BLEU_master import utils
from BLEU_master import precision
from BLEU_master import bleu_score
# from BLEU_master import demo

# %%
from BLEU_master.bleu_score import cal_corpus_bleu_score 

# %%
import numpy as np

def postprocess_text(preds, labels):

    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)  # input, output의 형태를 봐야함, 함수의 기능 

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing 
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    for p,l,i in zip(decoded_preds, decoded_labels, range(len(decoded_labels))):
        # if len(p) < 4 or len(t) < 4:
        # a b c -> ['a','b','c'] 
        # '' -> []
        if len(p.split(" ")) < 4:
            decoded_preds[i] = decoded_preds[i] + "@#$! @%^# @_#$ @#@@%"

        if len(l[0]) < 4:
            decoded_labels[i] = [decoded_labels[i][0] + "^@#$! @%^# @_#$ @#@@%"]  
 
        if len(p) <= 0:
            print('p 작음', p)
            decoded_preds[i] = decoded_preds[i] + "@#$! @%^# @_#$ @#@@%"

        if len(l[0]) <= 0:
            print('l 작음', l)
            decoded_labels[i] = [decoded_labels[i][0] + "^@#$! @%^# @_#$ @#@@%"]   # label 1개니까 0으로   

    result = cal_corpus_bleu_score(decoded_preds, decoded_labels,
                      weights=(0.25, 0.25, 0.25, 0.25), N=4)
    result = {"bleu": result}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


# %% [markdown]
# #### 2. 모델 
# - Fine-tuning, downstream task 
# - [Seq2SeqTrainer](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer)  사용

# %%
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# %% [markdown]
# #### 3. 학습 파라미터

# %%
# 하이퍼파라미터

batch_size = 32  # 모델 복잡도 조정 
args = Seq2SeqTrainingArguments(
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,  # 데이터셋 중간에 3번 저장 
    # num_train_epochs=20,  # epoch = 20로 돌려보기
    num_train_epochs=20 if use_cuda == True else 5,  # epoch = 20로 돌려보기
    predict_with_generate=True, # mixed precision; 모델 더 빨라짐 
    # fp16 = True,  # GPU에서 학습 속도를 높여줌
    fp16 = True if use_cuda == True else False,  # GPU에서 학습 속도를 높여줌
    output_dir="/output",  # 모델 예측값, checkpoint 저장할 디렉토리 추가 
    load_best_model_at_end = True,  # 가장 좋은 모델 저장
    save_strategy='epoch' # we cannot set it to "no". Otherwise, the model cannot guess the best checkpoint.
)


# %% [markdown]
# - early stopping = 3으로 설정

# %%
from transformers import EarlyStoppingCallback
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)
callbacks = [early_stopping_callback]

# %%
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=callbacks
)

# %% [markdown]
# ## 모델 학습

# %% [markdown]
# #### 1. 학습
# - 1000개 데이터로 학습

# %%
print("================== TRAINING MODEL =================")
now = datetime.datetime.now()
print("\t", now)

# %%
train_history = trainer.train()
print(train_history)

# %%
print("================== Save train history =================")

print(joblib.dump(train_history, 'train_history.pkl'))

# %% [markdown]
# #### 2. 모델 저장
# - model_checkpoint(모델 식별자) 대신에 내가 저장한 모델 경로 `saved_model = "./dart_pretrained_model"` 넣어주기

# %%
saved_model = "./dart_pretrained_model"

model.save_pretrained(saved_model)
tokenizer.save_pretrained(saved_model)

# %%
print("================== Save kobart-mini model, tokenizer ================")

print(joblib.dump(model, 'kobart_model.pkl'))
print(joblib.dump(tokenizer, 'kobart_tokenizer.pkl'))

# %% [markdown]
# > ## 모델 인퍼런스
# - [Huggingface Pipelines](https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/pipelines) 

# %% [markdown]
# #### 1. [Text2TextGenerationPipeline](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.Text2TextGenerationPipeline) 사용
# - max length = 128

# %%
from transformers import pipeline

dart_text2text = pipeline("text2text-generation", 
    saved_model, max_length=128)

# %% [markdown]
# #### 2. batch 32로 만들어서 model inference 진행 필요 -> GPU 메모리 부족
# - 32개씩 batch list로 묶여있는 애들을 하나로 합쳐줘야 함 ! 

# %%
print("================== MODEL INFERENCE =================")
now = datetime.datetime.now()
print("\t", now)

# %%
inf_batch_size = 32
train_pred_all = []
valid_pred_all = []

# train pred text 생성 
with open('train_pred.txt', 'w', encoding='UTF8') as f:
    for i in range(0, len(train_text), inf_batch_size):
        i_max = min(i+inf_batch_size, len(train_text))
        train_pred = dart_text2text(train_text[i:i_max])   
        print(f"=========== imax: {i_max} =========")    
        f.write(str(train_pred))
        train_pred_all += train_pred
    # print(f"=========== imax: {i_max} =========")    


# valid pred text 생성 
with open('val_pred.txt', 'w', encoding='UTF8') as f:
    for i in range(0, len(valid_text), inf_batch_size):
        i_max = min(i+inf_batch_size, len(valid_text))
        val_pred = dart_text2text(valid_text[i:i_max])
        print(f"=========== imax: {i_max} =========")    
        f.write(str(val_pred))
        valid_pred_all += val_pred
    # print(f"=========== imax: {i_max} =========")    


# %%
print(len(train_pred_all))
print(len(valid_pred_all))

# %% [markdown]
# > ## 모델 평가

# %% [markdown]
# #### 1. 딕셔너리의 string values를 리스트에 넣기

# %%
print("================== PREDICTION =================")
now = datetime.datetime.now()
print(now)

# %%
train_pred = [s['generated_text'] for s in train_pred_all if 'generated_text' in s]
val_pred = [s['generated_text'] for s in valid_pred_all if 'generated_text' in s]

# %%
print("================== Save train prediction, valid prediction =================")

print(joblib.dump(train_pred, 'train_prediction.pkl'))
print(joblib.dump(val_pred, 'valid_prediction.pkl'))

# %% [markdown]
# #### 2. compute metric 할 수 있도록 입력값 형태 맞추기
# - pred = ['string']
# - target = [['string'], ['string','string']]

# %%
# train data

pred_id = list(map(lambda x : tokenizer(x, max_length=128)['input_ids'], train_pred))
target_id = list(map(lambda x : tokenizer(x, max_length=128)['input_ids'], train_label_text))

pred_decode = tokenizer.batch_decode(pred_id, skip_special_tokens=True)
target_decode = tokenizer.batch_decode(target_id, skip_special_tokens=True)

pred_id_t =  [pred.strip() for pred in pred_decode]
target_id_t = [[label.strip()] for label in target_decode]


# %%
# valid data

pred_id = list(map(lambda x : tokenizer(x, max_length=128)['input_ids'], val_pred))
target_id = list(map(lambda x : tokenizer(x, max_length=128)['input_ids'], valid_label_text))

pred_decode = tokenizer.batch_decode(pred_id, skip_special_tokens=True)
target_decode = tokenizer.batch_decode(target_id, skip_special_tokens=True)

pred_id_v =  [pred.strip() for pred in pred_decode]
target_id_v = [[label.strip()] for label in target_decode]

# %%
print("================== FIND N <= 0 OR N < 4 =================")

print(pred_id_t[37], '\t num: ', len(pred_id_t[37]))
print(pred_id_t[74], '\t num: ', len(pred_id_t[74]))
print(pred_id_v[96], '\t num: ', len(pred_id_v[96]))

print(pred_id_t[36:37], '\t num: ', len(pred_id_t[36:37]))
print(pred_id_t[73:74], '\t num: ', len(pred_id_t[73:74]))
print(pred_id_v[95:96], '\t num: ', len(pred_id_v[95:96]))

# %%
for i in range(len(pred_id_t)):
    # print(i, "길이 : ", len(pred_id_t[i]))

    if len(pred_id_t[i]) <= 0:
        pred_id_t[i] += "@#$! @%^# @_#$ @#@@%"
        print(">>> train: ", pred_id_t[i])
        print(i, "번째 길이 : ", len(pred_id_t[i]))
        pred_id_t[i].split(" ")

for i in range(len(pred_id_v)):

    if len(pred_id_v[i]) <= 0:
        pred_id_v[i] += "@#$! @%^# @_#$ @#@@%"
        print(">>> valid: ", pred_id_v[i])
        print(i, "번째 길이 : ", len(pred_id_v[i]))
        pred_id_v[i].split(" ")

# %%
print("================== CHANGE SUCCEED =================")

print(pred_id_t[37], '\t num: ', len(pred_id_t[37]))
print(pred_id_t[74], '\t num: ', len(pred_id_t[74]))
print(pred_id_v[96], '\t num: ', len(pred_id_v[96]))

print(pred_id_t[36:37], '\t num: ', len(pred_id_t[36:37]))
print(pred_id_t[73:74], '\t num: ', len(pred_id_t[73:74]))
print(pred_id_v[95:96], '\t num: ', len(pred_id_v[95:96]))

# %%
print("======== Save train, val > prediction_id, target_id  ===========")

print(joblib.dump(pred_id_t, 'train_pred_id.pkl'))
print(joblib.dump(target_id_t, 'train_target_id.pkl'))
print(joblib.dump(pred_id_v, 'valid_pred_id.pkl'))
print(joblib.dump(target_id_v, 'valid_target_id.pkl'))

# %% [markdown]
# #### 3. bleu score로 모델 평가
# - [torch metrics](https://torchmetrics.readthedocs.io/en/v0.8.0/text/bleu_score.html)
# - [huggingface](https://huggingface.co/spaces/evaluate-metric/bleu)

# %%
print("================== BLEU SCORE =================")
now = datetime.datetime.now()
print("\t", now)

# %%
train_score = cal_corpus_bleu_score(pred_id_t, target_id_t,
                      weights=(0.25, 0.25, 0.25, 0.25), N=4)
print(train_score)

# %%
valid_score = cal_corpus_bleu_score(pred_id_v, target_id_v,
                      weights=(0.25, 0.25, 0.25, 0.25), N=4)
print(valid_score)

# %%
print("========= Save cal_corpus_bleu_score > train, valid ========")

print(joblib.dump(train_score, 'train_bleu_score.pkl'))
print(joblib.dump(valid_score, 'valid_bleu_score.pkl'))

# %% [markdown]
# #### 4. Trainer.evaluate 결과와 비교

# %%
print("================== EVALUATION =================")
now = datetime.datetime.now()
print("\t", now)

# %%
train_trainer = trainer.evaluate(eval_dataset=train_dataset, max_length=max_target_length)
print(train_trainer)

# %%
valid_trainer = trainer.evaluate(eval_dataset=val_dataset, max_length=max_target_length)
print(valid_trainer)

# %%
print("======== Save trainer evaluation score > train, valid =========")

print(joblib.dump(train_trainer, 'train_trainer_score.pkl'))
print(joblib.dump(valid_trainer, 'valid_trainer_score.pkl'))

# %% [markdown]
# > ## 예측값 저장 
# - prediction 통해 나온 텍스트

# %%
# Train data

with open('train_pred.txt', 'w', encoding='UTF8') as f:
    for line in train_pred:
        # line += '\n'
        f.writelines(f'{line}\n')

# valid data

with open('valid_pred.txt', 'w', encoding='UTF8') as f:
    for line in val_pred:
        f.writelines(f'{line}\n')

# %%
print("============ Load train prediction, valid prediction ============")

# print(joblib.load('train_prediction.pkl'))
# print(joblib.load('valid_prediction.pkl'))

# %%
print(train_pred[:3])
print(val_pred[:3])

# %%
print("================== DONE :-] ==================")
now = datetime.datetime.now()
print("\t", now)

# %% [markdown]
