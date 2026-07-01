# CO₂ Emission Prediction System (যানবাহনের কার্বন ডাই অক্সাইড নির্গমন পূর্বাভাস সিস্টেম)

একটি সুপারভাইজড মেশিন লার্নিং রিগ্রেশন প্রজেক্ট যা গাড়ির বিভিন্ন বৈশিষ্ট্য বিশ্লেষণ করে প্রতি কিলোমিটারে নির্গত কার্বন ডাই অক্সাইডের (CO₂ in g/km) পরিমাণ পূর্বাভাস দেয়।

---

## 🎯 Project Overview (প্রজেক্টের মূল উদ্দেশ্য)
যানবাহন থেকে নির্গত ক্ষতিকারক ধোঁয়া বায়ুমণ্ডলে গ্রিনহাউস গ্যাসের মাত্রা বাড়ায় যা বিশ্ব উষ্ণায়নের জন্য দায়ী। এই প্রজেক্টের উদ্দেশ্য হলো গাড়ির ইঞ্জিন সাইজ, সিলিন্ডার সংখ্যা, ট্রান্সমিশন, ফুয়েল টাইপ এবং জ্বালানি সাশ্রয়ের হার (Fuel Consumption) বিশ্লেষণ করে তার কার্বন নিঃসরণের হার নিখুঁতভাবে পূর্বাভাস করা। 
এটি পরিবেশ সচেতন ক্রেতা ও প্রস্তুতকারকদের পরিবেশ-বান্ধব গাড়ি নির্বাচনে এবং কার্বন ফুটপ্রিন্ট মূল্যায়নে সহায়তা করবে।

---

## 📁 Project Structure (প্রজেক্টের ফাইল কাঠামো)
- **`FuelConsumptionCo2.csv`**: কানাডা সরকারের জ্বালানি গ্রাস ও কার্বন নিঃসরণ সংক্রান্ত মূল ডেটাসেট (১০৬৭ টি গাড়ির ডেটা)।
- **`CO2_Emission_Prediction.ipynb`**: প্রজেক্টের জুপিটার নোটবুক যেখানে ডেটা লোড, EDA, প্রি-প্রসেসিং, মডেল ট্রেনিং ও ইভ্যালুয়েশন কোড রয়েছে।
- **`train.py`**: মডেল ট্রেইনিং ও তুলনা করার স্বয়ংক্রিয় পাইথন স্ক্রিপ্ট যা সেরা মডেলকে `model.pkl` হিসেবে সেভ করে।
- **`main.py`**: Flask ব্যাকএন্ড সার্ভার যা ওয়েব ফর্মের রিকোয়েস্ট নিয়ে ML মডেলের মাধ্যমে প্রেডিকশন দেয়।
- **`templates/`**:
  - `index.html`: ডার্ক মোড ও গ্লাস মরফিজম সহ প্রিমিয়াম ইনপুট ফর্ম।
  - `result.html`: রেডিয়াল গেজ ইন্ডিকেটর ও মডেল ডায়াগনস্টিক চার্ট সহ রেজাল্ট ড্যাশবোর্ড।
- **`static/images/`**: স্বয়ংক্রিয়ভাবে জেনারেট হওয়া ভিজ্যুয়ালাইজেশন ইমেজসমূহ।
  - `feature_importance.png`: কোন কোন ফিচার নিঃসরণে সবচেয়ে বেশি প্রভাব ফেলে।
  - `actual_vs_predicted.png`: মডেলের প্রেডিকশন বনাম বাস্তব ডেটার তুলনা।
  - `residuals.png`: মডেলের ভুলের (residuals) বিন্যাস।

---

## 📊 Dataset & Features (ডেটাসেটের ফিচারসমূহ)
মডেলটি ট্রেইন করতে মোট ৭টি স্বাধীন ফিচার (Independent Variables) ব্যবহার করা হয়েছে:
1. **Engine Size (ENGINESIZE):** ইঞ্জিনের ভলিউম (লিটার)।
2. **Cylinders (CYLINDERS):** ইঞ্জিনে সিলিন্ডারের সংখ্যা (৩ থেকে ১২)।
3. **Fuel Type (FUELTYPE):** জ্বালানির ধরণ:
   - `X` = Regular Gasoline (সাধারণ অকটেন/পেট্রোল)
   - `Z` = Premium Gasoline (প্রিমিয়াম অকটেন)
   - `D` = Diesel (ডিজেল)
   - `E` = Ethanol (E85 ইথানল)
4. **Vehicle Class (VEHICLECLASS):** গাড়ির শ্রেণী (যেমন: Compact, SUV, Sedan, Van ইত্যাদি)।
5. **Transmission (TRANSMISSION):** গিয়ার ট্রান্সমিশন সিস্টেম (Manual, Automatic, CVT ইত্যাদি)।
6. **Fuel Consumption City (FUELCONSUMPTION_CITY):** শহরে প্রতি ১০০ কিলোমিটারে জ্বালানি খরচ (L/100km)।
7. **Fuel Consumption Highway (FUELCONSUMPTION_HWY):** হাইওয়েতে প্রতি ১০০ কিলোমিটারে জ্বালানি খরচ (L/100km)।

* **Target Variable:** **CO2EMISSIONS** (g/km) - প্রতি কিলোমিটারে কার্বন নির্গমন।

---

## 🔬 Model Performance & Comparison (মডেলসমূহের কার্যকারিতা তুলনা)
ডেটাসেটটিকে ৮০:২০ অনুপাতে ট্রেইন ও টেস্ট সেটে ভাগ করে ৫টি রিগ্রেশন অ্যালগরিদম তুলনা করা হয়েছে:

| Regression Model | MAE (g/km) | RMSE (g/km) | R² Score (R-squared) | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost Regressor (Best)** | **1.656** | **3.556** | **0.99694** | **0.711%** |
| **Random Forest** | 2.152 | 5.464 | 0.99278 | 0.928% |
| **Decision Tree** | 2.395 | 4.353 | 0.99542 | 0.991% |
| **Linear Regression** | 3.569 | 6.592 | 0.98949 | 1.413% |
| **Ridge Regression** | 3.813 | 6.596 | 0.98948 | 1.529% |

### Key Findings (প্রধান পর্যবেক্ষণসমূহ):
- **XGBoost Regressor** সবচেয়ে নিখুঁত ফলাফল দেখিয়েছে, যার **R² Score = 99.69%** এবং গড় ত্রুটি মাত্র **1.65 g/km** (MAPE = 0.711%)।
- ডিসিশন ট্রি ও র‍্যান্ডম ফরেস্টের মতো বৃক্ষ-ভিত্তিক অ্যালগরিদমগুলো সরল লিনিয়ার রিগ্রেশনের চেয়ে ভালো ফলাফল প্রদান করে কারণ গাড়ির ফিচারগুলোর সম্পর্ক পুরোপুরি সরলরৈখিক নয়।
- কার্বন নিঃসরণে সবচেয়ে বড় ভূমিকা রাখে **Fuel Consumption** এবং **Engine Size**।

---

## 🚀 How to Run the Project (কীভাবে প্রজেক্টটি চালাবেন)

### ১. লাইব্রেরিসমূহ ইন্সটল করুন:
প্রজেক্টের মেইন ফোল্ডারে গিয়ে কমান্ড লাইনে নিচের কমান্ডটি লিখুন:
```bash
pip install -r requirements.txt
```

### ২. মডেলটি পুনরায় ট্রেইন করুন (ঐচ্ছিক):
মডেল ট্রেইনিং স্ক্রিপ্টটি চালাতে এবং নতুন গ্রাফ তৈরি করতে লিখুন:
```bash
python train.py
```
*(এটি সফলভাবে শেষ হলে `model.pkl` এবং `static/images/` ফোল্ডারে ৩টি চার্ট তৈরি হবে)*

### ৩. Flask ওয়েব অ্যাপ্লিকেশন চালু করুন:
ব্যাকএন্ড সার্ভার চালু করতে লিখুন:
```bash
python main.py
```

### ৪. ব্রাউজারে রান করুন:
সার্ভার চালু হলে যেকোনো ওয়েব ব্রাউজারে গিয়ে নিচের লিঙ্কে প্রবেশ করুন:
`http://127.0.0.1:5000/`

---

## 💻 Web App Interface Features (ওয়েব অ্যাপের মূল সুবিধাসমূহ)
- **Interactive Form:** স্লাইডারের মাধ্যমে ইঞ্জিন সাইজ এবং জ্বালানি খরচ নিয়ন্ত্রণ করা যায় ও ড্রপডাউন দিয়ে ফুয়েল ও ট্রান্সমিশন নির্বাচন করা যায়।
- **Radial Gauge Rating:** প্রেডিকশন রেজাল্ট অনুযায়ী গাড়ির কার্বন লেভেল **Low** (সবুজ), **Moderate** (হলুদ) নাকি **High** (লাল) তা একটি বৃত্তাকার মিটারে রিয়েল-টাইমে কালার কোড সহ দেখায়।
- **Model Insights Dashboard:** ওয়েব অ্যাপের ভেতর থেকেই ব্যবহারকারীগণ ট্রেইন হওয়া সেরা মডেলটির **Feature Importance**, **Actual vs Predicted** এবং **Residual Plot** সরাসরি ট্যাব অপশনের মাধ্যমে দেখতে পারেন।

---

## 👥 Team Members (দলের সদস্যবৃন্দ)

> **Course:** Artificial Intelligence Lab | **Problem Type:** Supervised Regression

| # | Name | Role | Key Contributions |
| :---: | :--- | :--- | :--- |
| 1 | ⭐ **Tahmidul Alam Ahad** | Backend & ML Engineer | Flask web app (`main.py`), ML model pipeline, training & evaluation script (`train.py`), model serialization (`model.pkl`), end-to-end system integration |
| 2 | ⭐ **Abdur Rahman** | Backend & Data Engineer | Dataset preparation (`prepare_real_dataset.py`), preprocessing pipeline (StandardScaler, OneHotEncoder), Flask routing & form handling, CO₂ classification logic |
| 3 | **Tawsif Hossen** | Frontend Developer | HTML/CSS design for input form (`index.html`), UI layout and styling, interactive sliders and dropdown elements |
| 4 | **S.M Sayem** | Frontend & Tester | Result page design (`result.html`), radial gauge indicator, model diagnostics chart tabs, UI testing and debugging |
| 5 | **Mohammad Tareq Aziz** | Research & Docs | Dataset research and selection, project documentation (`README.md`), report writing, model performance analysis |

> ⭐ **Primary Contributors:** Tahmidul Alam Ahad and Abdur Rahman led the core backend and machine learning development.
