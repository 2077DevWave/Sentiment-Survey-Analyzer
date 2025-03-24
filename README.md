# Persian Sentiment Analyzer for Product Reviews
[![فارسی](https://img.shields.io/badge/زبان-فارسی-blue.svg)](README.fa.md)


## 📌 Overview
A sentiment analysis tool for Persian product reviews, classifying feedback into:
- ✅ Recommended 
- ❌ Not Recommended  
- 🤷 No Idea 

## ✨ Features
- **Preprocessing Pipeline**  
  Normalization, tokenization, stemming, and stopword removal for Persian text.
- **Word2Vec Embeddings**  
  Custom-trained embeddings on 150K+ Persian reviews.
- **Logistic Regression Classifier**  
  Achieves **75% accuracy** on large datasets.
- **Easy Prediction API**  
  `predict_recommendation()` function for new reviews.

## 📂 Dataset

I obtained a dataset of 150,000 samples from [quera.org](https://quera.org). Additionally, I acquired a larger dataset from [Kaggle](https://www.kaggle.com/datasets/radeai/digikala-comments-and-products) and included it in the repository as `big_train.csv`.

### Loading the Dataset

To train the model using the smaller dataset:
```python
train_data = pd.read_csv('train.csv')
```

For systems with powerful hardware to process the larger dataset:

``` bash
train_data = pd.read_csv('big_train.csv', usecols=['body', 'recommendation_status'])
```

This combined dataset merges samples from both Quera and Kaggle sources.
> Due to the large size of the `big_train.csv` file, I've split it into 12 RAR parts. To obtain the complete file, you only need to extract one of the parts.

Hardware Specifications
The project was executed on hardware with:

- 2 vCPU cores

- 5GB RAM

  If you encounter issues with library installation or hardware requirements, you can use [DeepNote](deepnote.com) as an alternative platform.

## 🛠️ Installation

1- Clone the repo:
``` bash
git clone https://github.com/RezaGooner/Sentiment-Survey-Analyzer.git
```
2- Install dependencies:
``` bash
pip install -r requirements.txt
```

## 🚀 Usage

``` bash
from model import predict_recommendation

result = predict_recommendation("این گوشی ارزش خرید دارد")
print(result)
```

---
## 📊 Performance
Dataset Size	Accuracy
150K (Quera)	60%
450K (Combined)	75%

---

## Real-World Test Cases
Below are sample reviews with star ratings, user recommendations, and model predictions for performance evaluation:



| Review | Stars | User Recommended | Model Prediction | TP | TN | FP | FN |
|--------|-------|------------------|------------------|----|----|----|----|
| با توجه به قیمت بهترین انتخابه | ⭐⭐⭐ | ✅ | Recommended | 1 | 0 | 0 | 0 |
| کیفیت خوبی داره و برای پوست صورت گزینه مناسبی هست و از خریدم راضی هستم | ⭐⭐⭐⭐⭐ | ✅ | Recommended | 1 | 0 | 0 | 0 |
| عالیه.حیف که گرون شد | ⭐⭐⭐ | ➖ | Recommended | 0 | 0 | 1 | 0 |
| دستمال خوب و به صرفه ای میباشد .اما کیفیت آن نسبت به جعبه ای یک مقدار خشک است ولی تو شرایط بازار الان به صرفه میباشد. یک مقدار خشک میباشد | ⭐⭐⭐ | ➖ | No Opinion | 0 | 0 | 0 | 0 |
| با جلد پاره ارسال شد! اونم وسیله بهداشتی | ⭐ | ❌ | Not Recommended | 0 | 1 | 0 | 0 |
| قیمت رو دسمال ۱۴۰۰۰ تومن تولیدبرای هفته پیشه بعد ۱۷۰۰۰ میفروشن متاسفم | ⭐ | ❌ | Not Recommended | 0 | 1 | 0 | 0 |
| بنده چند بار خرید کردم ولی این بار از بغل پاره شده بود و دستمال های داخل آن کثیف بود که برای یک کالای بهداشتی اصلا خوب نیست. فقط نمیدانم دیجی کالا اینو فرستاده یا وقتی ارسال شده پاره شده . ارزش مرجوع کردن هم نداشت و بعد از بیست روز به دستم رسید.. | ⭐⭐ | ➖ | Not Recommended | 0 | 1 | 0 | 0 |
| کیفیت دستمال را کم کردن به همین علت تخفیف میدن اگر خرید قبلی دارید کیفیت جنس دستمال را مقایسه کنید متوجه میشید | ⭐ | ➖ | Not Recommended | 0 | 1 | 0 | 0 |
| کاش قیمت پایین تر بود .ولی محصولات تنو از کیفیت خوبی برخوردار هستند | ⭐⭐⭐⭐ | ✅ | No Opinion | 0 | 0 | 0 | 1 |
| بالاتر از قیمت مصرف کننده بود و تولید برج 2 نخرید بیرون ارزون تره | ⭐ | ❌ | Not Recommended | 0 | 1 | 0 | 0 |

**Evaluation Metrics**:
- **Precision**: 100% (2/2 correct positive predictions)
- **Recall**: 66.7% (2/3 actual positives caught)
- **Specificity**: 100% (5/5 correct negative predictions)
- **No-Opinion Accuracy**: 50% (1 correct out of 2 neutral cases)

**Legend**:
- ✅ = User recommended (y)
- ❌ = User not recommended (n)
- ➖ = No opinion (-)
- TP = True Positive (True Recommended)
- TN = True Negative
- FP = False Positive
- FN = False Negative

---
## 🤝 Contribution
Pull requests welcome! For major changes, please open an [issue](https://github.com/RezaGooner/Sentiment-Survey-Analyzer/issues) first.

---
> Github.com/RezaGooner
