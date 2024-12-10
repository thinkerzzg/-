# 一、数据加载及类型转换
import polars as pl
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# 1. 数据加载
def set_table_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    for col in df.columns:
        if col[-1] in ("P", "A"):
            df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))
    return df

train_basetable = pl.read_csv(r"C:\Users\13256\Desktop\大数据\大作业\train_base.csv", ignore_errors=True)
train_static = pl.concat([
    pl.read_csv(r"C:\Users\13256\Desktop\大数据\大作业\train_static_0_0.csv", ignore_errors=True).pipe(set_table_dtypes),
    pl.read_csv(r"C:\Users\13256\Desktop\大数据\大作业\train_static_0_1.csv", ignore_errors=True).pipe(set_table_dtypes),
], how="vertical_relaxed")
train_static_cb = pl.read_csv(r"C:\Users\13256\Desktop\大数据\大作业\train_static_cb_0.csv", ignore_errors=True).pipe(set_table_dtypes)
train_person_1 = pl.read_csv(r"C:\Users\13256\Desktop\大数据\大作业\train_person_1.csv", ignore_errors=True).pipe(set_table_dtypes)
train_credit_bureau_b_2 = pl.read_csv(r"C:\Users\13256\Desktop\大数据\大作业\train_credit_bureau_b_2.csv", ignore_errors=True).pipe(set_table_dtypes)

test_basetable = pl.read_csv(r"C:\Users\13256\Desktop\大数据\大作业\test_base.csv", ignore_errors=True)
test_static = pl.concat(
    [
        pl.read_csv(r"C:\Users\13256\Desktop\大数据\大作业\test_static_0_0.csv", ignore_errors=True).pipe(set_table_dtypes),
        pl.read_csv(r"C:\Users\13256\Desktop\大数据\大作业\test_static_0_1.csv", ignore_errors=True).pipe(set_table_dtypes),
        pl.read_csv(r"C:\Users\13256\Desktop\大数据\大作业\test_static_0_2.csv", ignore_errors=True).pipe(set_table_dtypes),
    ],
    how="vertical_relaxed",
)
test_static_cb = pl.read_csv(r"C:\Users\13256\Desktop\大数据\大作业\test_static_cb_0.csv", ignore_errors=True).pipe(set_table_dtypes)
test_person_1 = pl.read_csv(r"C:\Users\13256\Desktop\大数据\大作业\test_person_1.csv", ignore_errors=True).pipe(set_table_dtypes)
test_credit_bureau_b_2 = pl.read_csv(r"C:\Users\13256\Desktop\大数据\大作业\test_credit_bureau_b_2.csv", ignore_errors=True).pipe(set_table_dtypes)
# 二、数据总览

# 1. 观察数据结构
# 查看数据的基本信息，如行列数、列名等
print(test_static_cb.shape)
print(test_static_cb.columns)

# 2. 观察数据的相关统计量
# 获取数据的基本统计信息
print(test_static_cb.describe())

# 3. 观察数据类型
# 查看每列的数据类型
print(test_static_cb.dtypes)
"""
# （三） 数据预处理

# 1. 查看数据缺失情况
# 检查每列缺失值的数量
missing_data = data.isnull().sum()
print("缺失值情况：")
print(missing_data)

# 可视化缺失值
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# 处理缺失值，可以选择填充或删除
# 例如，填充缺失值为每列的中位数
data_filled = data.fillna(data.median())

# 2. 查看数据异常情况
# 使用 Z-score 检测异常值
z_scores = np.abs(stats.zscore(data.select_dtypes(include=['float64', 'int64'])))
outliers = (z_scores > 3).any(axis=1)  # 标记 Z-score 大于3的行为异常值
print("异常值行索引：", data[outliers].index)

# 可视化异常值
for col in data.select_dtypes(include=['float64', 'int64']):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data[col])
    plt.title(f'Boxplot for {col}')
    plt.show()

# （四） 数据特征分析

# 1. 预测值分布
# 假设目标变量是 'target' 列
plt.figure(figsize=(10, 6))
sns.histplot(data['target'], kde=True)
plt.title('Target Variable Distribution')
plt.show()

# 2. 类别特征分析
# 查看类别特征的频率分布
categorical_features = data.select_dtypes(include=['object'])
for col in categorical_features.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(y=data[col])
    plt.title(f'Frequency Distribution of {col}')
    plt.show()

# 类别特征的编码（例如，使用 One-Hot 编码）
data_encoded = pd.get_dummies(data, drop_first=True)

# 3. 数字特征分析
# 查看数字特征的基本统计信息
numeric_features = data.select_dtypes(include=['float64', 'int64'])
print("数字特征的基本统计信息：")
print(numeric_features.describe())

# 查看数值特征的相关性
plt.figure(figsize=(12, 8))
correlation_matrix = numeric_features.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# 可视化数值特征的分布
for col in numeric_features.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
"""
# 二、特征工程

# 1. 聚合操作和特征提取
train_person_1_feats_1 = train_person_1.group_by("case_id").agg(
    pl.col("mainoccupationinc_384A").max().alias("mainoccupationinc_384A_max"),
    (pl.col("incometype_1044T") == "SELFEMPLOYED").max().alias("mainoccupationinc_384A_any_selfemployed")
)
train_person_1_feats_2 = train_person_1.select(["case_id", "num_group1", "housetype_905L"]).filter(
    pl.col("num_group1") == 0
).drop("num_group1").rename({"housetype_905L": "person_housetype"})
train_credit_bureau_b_2_feats = train_credit_bureau_b_2.group_by("case_id").agg(
    pl.col("pmts_pmtsoverdue_635A").max().alias("pmts_pmtsoverdue_635A_max"),
    (pl.col("pmts_dpdvalue_108P") > 31).max().alias("pmts_dpdvalue_108P_over31")
)

# 2. 特征列选择和数据合并
selected_static_cols = [col for col in train_static.columns if col[-1] in ("A", "M")]
selected_static_cb_cols = [col for col in train_static_cb.columns if col[-1] in ("A", "M")]
data = train_basetable.join(
    train_static.select(["case_id"] + selected_static_cols), how="left", on="case_id"
).join(
    train_static_cb.select(["case_id"] + selected_static_cb_cols), how="left", on="case_id"
).join(
    train_person_1_feats_1, how="left", on="case_id"
).join(
    train_person_1_feats_2, how="left", on="case_id"
).join(
    train_credit_bureau_b_2_feats, how="left", on="case_id"
)

# 三、数据划分和转换
case_ids = data["case_id"].unique().shuffle(seed=1)
case_ids_train, case_ids_test = train_test_split(case_ids, train_size=0.6, random_state=1)
case_ids_valid, case_ids_test = train_test_split(case_ids_test, train_size=0.5, random_state=1)
cols_pred = [col for col in data.columns if col[-1].isupper() and col[:-1].islower()]

# 将 Polars DataFrame 转换为 Pandas
def from_polars_to_pandas(case_ids: pl.DataFrame):
    return (
        data.filter(pl.col("case_id").is_in(case_ids))[["case_id", "WEEK_NUM", "target"]].to_pandas(),
        data.filter(pl.col("case_id").is_in(case_ids))[cols_pred].to_pandas(),
        data.filter(pl.col("case_id").is_in(case_ids))["target"].to_pandas()
    )

base_train, X_train, y_train = from_polars_to_pandas(case_ids_train)
base_valid, X_valid, y_valid = from_polars_to_pandas(case_ids_valid)
base_test, X_test, y_test = from_polars_to_pandas(case_ids_test)

# 转换类别特征
def convert_strings(df: pd.DataFrame):
    for col in df.columns:
        if df[col].dtype.name in ['object', 'string']:
            df[col] = df[col].astype("string").astype('category')
            current_categories = df[col].cat.categories
            new_categories = current_categories.to_list() + ["Unknown"]
            new_dtype = pd.CategoricalDtype(categories=new_categories, ordered=True)
            df[col] = df[col].astype(new_dtype)
    return df

for df in [X_train, X_valid, X_test]:
    df = convert_strings(df)

# 四、LightGBM 模型训练
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train)

# 定义模型参数
params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "max_depth": 3,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "n_estimators": 1000,
    "verbose": -1,
}

# 模型训练
gbm = lgb.train(
    params,
    lgb_train,
    valid_sets=lgb_valid,
    callbacks=[lgb.log_evaluation(50), lgb.early_stopping(10)]
)

"""
# 定义随机森林模型
rf = RandomForestClassifier(
    n_estimators=200,     # 决策树数量
    max_depth=15,         # 最大树深度
    min_samples_split=5,  # 最小分裂样本数
    min_samples_leaf=3,   # 最小叶子样本数
    random_state=42,      # 随机种子
    n_jobs=-1             # 使用所有 CPU 核心
)

# 训练模型
rf.fit(X_train, y_train)
"""
# 五、模型评估
# 对训练、验证和测试集分别预测并计算 AUC
for base, X in [(base_train, X_train), (base_valid, X_valid), (base_test, X_test)]:
    y_pred = gbm.predict(X, num_iteration=gbm.best_iteration)  # 使用最佳迭代次数进行预测
    base["score"] = y_pred  # 将预测结果存入数据集

# 打印每个数据集的 AUC 分数
print(f'The AUC score on the train set is: {roc_auc_score(base_train["target"], base_train["score"])}')
print(f'The AUC score on the valid set is: {roc_auc_score(base_valid["target"], base_valid["score"])}')
print(f'The AUC score on the test set is: {roc_auc_score(base_test["target"], base_test["score"])}')


# 定义 gini_stability 函数
def gini_stability(base, w_fallingrate=88.0, w_resstd=-0.5):
    """
    计算基尼稳定性指标，包括基尼均值、下降率惩罚、残差惩罚。
    """
    # 按周计算基尼系数
    gini_in_time = (
        base.loc[:, ["WEEK_NUM", "target", "score"]]
        .sort_values("WEEK_NUM")
        .groupby("WEEK_NUM")[["target", "score"]]
        .apply(lambda x: 2 * roc_auc_score(x["target"], x["score"]) - 1)
        .tolist()
    )

    # 创建 x 和 y，用于线性回归
    x = np.arange(len(gini_in_time))  # 时间序列
    y = gini_in_time  # 基尼系数
    a, b = np.polyfit(x, y, 1)  # 拟合线性回归，得到斜率 a 和截距 b

    # 计算残差和标准差
    y_hat = a * x + b  # 预测值
    residuals = y - y_hat  # 残差
    res_std = np.std(residuals)  # 残差标准差

    # 计算平均基尼和稳定性分数
    avg_gini = np.mean(gini_in_time)  # 基尼均值
    falling_rate = min(0, a)  # 斜率下降惩罚
    stability_score = avg_gini + w_fallingrate * falling_rate + w_resstd * res_std  # 最终得分

    return stability_score


# 分别计算每个数据集的稳定性分数
stability_score_train = gini_stability(base_train)
stability_score_valid = gini_stability(base_valid)
stability_score_test = gini_stability(base_test)

# 打印稳定性分数
print(f'The stability score on the train set is: {stability_score_train}')
print(f'The stability score on the valid set is: {stability_score_valid}')
print(f'The stability score on the test set is: {stability_score_test}')

# test
test_person_1_feats_1 = test_person_1.group_by("case_id").agg(
    pl.col("mainoccupationinc_384A").max().alias("mainoccupationinc_384A_max"),
    (pl.col("incometype_1044T") == "SELFEMPLOYED").max().alias("mainoccupationinc_384A_any_selfemployed")
)

test_person_1_feats_2 = test_person_1.select(["case_id", "num_group1", "housetype_905L"]).filter(
    pl.col("num_group1") == 0
).drop("num_group1").rename({"housetype_905L": "person_housetype"})

test_credit_bureau_b_2_feats = test_credit_bureau_b_2.group_by("case_id").agg(
    pl.col("pmts_pmtsoverdue_635A").max().alias("pmts_pmtsoverdue_635A_max"),
    (pl.col("pmts_dpdvalue_108P") > 31).max().alias("pmts_dpdvalue_108P_over31")
)

# 选择特征列
selected_static_cols = [col for col in test_static.columns if col[-1] in ("A", "M")]
selected_static_cb_cols = [col for col in test_static_cb.columns if col[-1] in ("A", "M")]

data_submission = test_basetable.join(
    test_static.select(["case_id"] + selected_static_cols), how="left", on="case_id"
).join(
    test_static_cb.select(["case_id"] + selected_static_cb_cols), how="left", on="case_id"
).join(
    test_person_1_feats_1, how="left", on="case_id"
).join(
    test_person_1_feats_2, how="left", on="case_id"
).join(
    test_credit_bureau_b_2_feats, how="left", on="case_id"
)

X_submission = data_submission[cols_pred].to_pandas()
X_submission = convert_strings(X_submission)
categorical_cols = X_train.select_dtypes(include=['category']).columns

for col in categorical_cols:
    train_categories = set(X_train[col].cat.categories)
    submission_categories = set(X_submission[col].cat.categories)
    new_categories = submission_categories - train_categories
    X_submission.loc[X_submission[col].isin(new_categories), col] = "Unknown"
    new_dtype = pd.CategoricalDtype(categories=train_categories, ordered=True)
    X_train[col] = X_train[col].astype(new_dtype)
    X_submission[col] = X_submission[col].astype(new_dtype)

y_submission_pred = gbm.predict(X_submission, num_iteration=gbm.best_iteration)

submission = pd.DataFrame({
    "case_id": data_submission["case_id"].to_numpy(),
    "score": y_submission_pred
}).set_index('case_id')
submission.to_csv("./submission.csv")