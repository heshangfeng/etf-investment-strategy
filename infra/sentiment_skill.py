"""
ETF 智能投资分析系统 - FinBERT 金融情感分析

替代 SnowNLP，使用预训练中文金融情感模型。
模型: uer/roberta-base-finetuned-chinanews-chinese
"""
import os
import logging
from typing import Optional

FINBERT_MODEL = os.getenv("FINBERT_MODEL", "uer/roberta-base-finetuned-chinanews-chinese")


class FinBertSentiment:
    """FinBERT 金融情感分析器。首次加载会下载模型（约 400MB）。"""

    _pipeline = None

    @classmethod
    def _ensure_model(cls):
        """延迟加载模型（仅首次调用时下载）。"""
        if cls._pipeline is None:
            try:
                from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
                from transformers.utils import logging as tf_logging
                tf_logging.set_verbosity_error()

                tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
                model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
                cls._pipeline = pipeline(
                    "sentiment-analysis",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=512,
                    truncation=True,
                )
            except Exception as e:
                logging.warning(f"FinBERT 模型加载失败: {e}，回退到中性评分")
                return False
        return True

    @classmethod
    def analyze(cls, text: str) -> tuple[float, str]:
        """
        情感分析。

        Args:
            text: 输入文本

        Returns:
            (得分 0-100, 标签 positive/negative)
        """
        if not text or len(text.strip()) < 5:
            return 50.0, "neutral"

        if not cls._ensure_model():
            return 50.0, "neutral"

        try:
            result = cls._pipeline(text[:512])[0]
            label = result["label"]
            score = result["score"]

            # 转成 0-100 分制
            if label.upper() == "POSITIVE" or "积极" in label or "正面" in label:
                normalized = 50 + score * 50  # 50~100
                tag = "positive"
            elif label.upper() == "NEGATIVE" or "消极" in label or "负面" in label:
                normalized = 50 - score * 50  # 0~50
                tag = "negative"
            else:
                normalized = 50.0
                tag = "neutral"

            return round(float(normalized), 2), tag
        except Exception as e:
            logging.warning(f"FinBERT 分析失败: {e}")
            return 50.0, "neutral"
