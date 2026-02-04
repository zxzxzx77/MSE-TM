

# ============================================
# 1. 
# ============================================
!pip install -q bertopic sentence-transformers datasets nltk gensim scikit-learn umap-learn ripser

# ============================================
# 2. 
# ============================================
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
import umap
import re

from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cdist
from scipy.stats import entropy

print("? Libraries are ready")

# ============================================
# 3. Mathematical Equations Tracker
# ============================================
class EquationTracker:
    """
    
    """
    
    def __init__(self):
        self.equations_applied = {
            'DQS': [],
            'SDW': [],
            'TF-IDF-Enhanced': [],
            'Positional-Weight': []
        }
    
    def record_dqs(self, value: float):
        self.equations_applied['DQS'].append(value)
    
    def record_sdw(self, value: float):
        self.equations_applied['SDW'].append(value)
    
    def record_tfidf_enhanced(self, value: float):
        self.equations_applied['TF-IDF-Enhanced'].append(value)
    
    def record_positional(self, value: float):
        self.equations_applied['Positional-Weight'].append(value)
    
    def print_report(self):
        print("\n" + "="*70)
        
        print("?? Report on the application of mathematical equations")
        print("="*70)
        
        for eq_name, values in self.equations_applied.items():
            if len(values) > 0:
                print(f"\n? {eq_name}:")
                print(f"   عدد مرات التطبيق: {len(values)}")
                print(f"   المتوسط: {np.mean(values):.4f}")
                print(f"   الانحراف المعياري: {np.std(values):.4f}")
        
        print("\n" + "="*70)

# إنشاء tracker عام
eq_tracker = EquationTracker()

# ============================================
# 4. Download data 
# ============================================
class OptimizedPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['subject', 'organization', 'lines', 'writes', 
                               'article', 'like', 'would', 'could', 'one', 'get',
                               'also', 'may', 'said', 'say', 'use', 'well', 'know'])
    
    def clean_20newsgroup(self, text):
        text = re.sub(r'From:.*\n|Subject:.*\n|Organization:.*\n|Lines:.*\n', '', text)
        text = re.sub(r'Distribution:.*\n|NNTP-Posting-Host:.*\n|X-.*\n|Reply-To:.*\n', '', text)
        text = re.sub(r'\S+@\S+|http\S+|www\S+|ftp\S+', ' ', text)
        text = re.sub(r'\|>.*\n|>>.*\n', ' ', text)
        text = re.sub(r'\b\d{1,3}\b', ' ', text)
        text = re.sub(r'[^\w\s-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower().strip()
        
        words = text.split()
        words = [w for w in words if w not in self.stop_words and 3 <= len(w) <= 20]
        
        return ' '.join(words)
    
    def load_20newsgroup(self):
        print("Download data 20NewsGroup...")
        newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
        
        print("جاري التنظيف...")
        cleaned = []
        for t in tqdm(newsgroups.data):
            c = self.clean_20newsgroup(t)
            cleaned.append(c if len(c) > 100 else '')
        
        valid_idx = [i for i, t in enumerate(cleaned) if len(t) > 100 and len(t.split()) >= 20]
        
        texts = [cleaned[i] for i in valid_idx]
        labels = [newsgroups.target[i] for i in valid_idx]
        
        print(f"? {len(texts)} وثيقة، {len(set(labels))} فئة")
        return texts, labels, newsgroups.target_names

# ============================================
# 5. المعادلة 1: Document Quality Scoring
# ============================================
class DocumentQualityScorer:
    """
    ???????????????????????????????????????????????????????????????????
    المعادلة الرياضياتية 1: Document Quality Score (DQS)
    ???????????????????????????????????????????????????????????????????
    
    mathematical equation:
    ????????????????????
    DQS(d) = ?·LD(d) + ?·SC(d) + ?·LP(d)
    
    حيث:
    - LD(d) = |unique_words| / |total_words|        (Lexical Diversity)
    - SC(d) = avg_cosine_sim(sentences)              (Semantic Coherence)
    - LP(d) = 1 / (1 + e^(-k·(|d| - ?)))            (Length Penalty - Sigmoid)
    - ? = 0.4, ? = 0.4, ? = 0.2                     (Weights)
    - k = 0.02, ? = 150                              (Sigmoid parameters)
    
  
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.optimal_length = 150  # ? في المعادلة
        self.k = 0.02              # k في المعادلة
    
    def score_document(self, text: str) -> float:
        """
        تطبيق المعادلة DQS خطوة بخطوة
        """
        words = text.split()
        
        if len(words) < 20:
            return 0.0
        
        # ?????????????????????????????????????????????????????????????
        # الخطوة 1: حساب Lexical Diversity (LD)
        # ?????????????????????????????????????????????????????????????
        # المعادلة: LD = |unique_words| / |total_words|
        unique_words = set(words)
        lexical_div = len(unique_words) / len(words)
        #             ???????????????   ?????????????
        #              |unique|        |total|
        
        # ?????????????????????????????????????????????????????????????
        # الخطوة 2: حساب Semantic Coherence (SC)
        # ?????????????????????????????????????????????????????????????
        # المعادلة: SC = (1/n²) ?? ?? cos_sim(s?, s?)
        sentences = sent_tokenize(text)
        if len(sentences) >= 3:
            try:
                sent_embeddings = self.embedding_model.encode(sentences[:10])
                similarities = cosine_similarity(sent_embeddings)
                #              ???????????????????????
                #                    cos_sim matrix
                
                mask = np.ones_like(similarities, dtype=bool)
                np.fill_diagonal(mask, False)
                semantic_coh = similarities[mask].mean()
                #              ???????????????
                #                  avg similarity
            except:
                semantic_coh = 0.5
        else:
            semantic_coh = 0.5
        
        # ?????????????????????????????????????????????????????????????
        #  3:  Length Penalty (LP) - Sigmoid
        # ?????????????????????????????????????????????????????????????
        # المعادلة: LP = 1 / (1 + e^(-k·(L - ?)))
        length = len(words)
        exponent = -self.k * (length - self.optimal_length)
        #           ???????   ???????????????????
        #             k      (length - ?)
        
        length_pen = 1 / (1 + np.exp(exponent))
        #            ???????????????????????
        #                   sigmoid
        
        # ?????????????????????????????????????????????????????????????
        # الخطوة 4: دمج المعادلة الكاملة
        # ?????????????????????????????????????????????????????????????
        # المعادلة النهائية: DQS = ?·LD + ?·SC + ?·LP
        alpha, beta, gamma = 0.4, 0.4, 0.2
        dqs = alpha * lexical_div + beta * semantic_coh + gamma * length_pen
        #     ?????   ?????????????   ?????   ???????????????   ?????   ?????????????
        #       ?          LD           ?          SC           ?         LP
        
        # ?? تسجيل تطبيق المعادلة
        eq_tracker.record_dqs(dqs)
        
        return float(dqs)
    
    def filter_corpus(self, texts: List[str], labels: List[int], 
                     keep_percentile: float = 75.0) -> Tuple:
        print(f"\n?? تطبيق المعادلة 1: Document Quality Scoring...")
        
        scores = []
        for text in tqdm(texts, desc="Computing DQS"):
            score = self.score_document(text)
            scores.append(score)
        
        scores = np.array(scores)
        threshold = np.percentile(scores, 100 - keep_percentile)
        
        keep_idx = scores >= threshold
        
        filtered_texts = [texts[i] for i in range(len(texts)) if keep_idx[i]]
        filtered_labels = [labels[i] for i in range(len(labels)) if keep_idx[i]]
        
        print(f"? DQS: استبعدنا {np.sum(~keep_idx)} وثيقة منخفضة الجودة")
        print(f"? تبقى {len(filtered_texts)} وثيقة (متوسط DQS: {scores[keep_idx].mean():.4f})")
        
        return filtered_texts, filtered_labels

# ============================================
# 6. المعادلة 2: Semantic Density Weighting
# ============================================
class SemanticDensityWeighter:
    """
    ???????????????????????????????????????????????????????????????????
    المعادلة الرياضياتية 2: Semantic Density Weighting (SDW)
    ???????????????????????????????????????????????????????????????????
    
    
    
    الصيغة الرياضياتية:
    ????????????????????
    SDW(chunk) = (1/|words|) ?? [1 + ln(1 + ?_local(w?))]
    
    حيث:
    - ?_local(w) = |context_window| × (|unique_words| / |total_words|)
    - chunk = مجموعة من 15-25 كلمة متتالية
    ???????????????????????????????????????????????????????????????????
    """
    
    def __init__(self, embedding_model, chunk_size: int = 20, window_size: int = 5):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size  # عدد الكلمات في كل chunk
        self.window_size = window_size
    
    def split_into_chunks(self, text: str) -> List[str]:
      
        words = text.split()
        chunks = []
        
        if len(words) <= self.chunk_size:
            return [text]
        
        # تقسيم إلى chunks متداخلة (overlap = 50%)
        step = max(1, self.chunk_size // 2)
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) >= 10:  # chunks صغيرة جداً نتجاهلها
                chunks.append(' '.join(chunk_words))
        
        return chunks
    
    def compute_chunk_weights(self, chunks: List[str]) -> Dict[int, float]:
        """
        حساب SDW لكل chunk
        """
        weights = {}
        
        for i, chunk in enumerate(chunks):
            words = chunk.split()
            
            if len(words) < 5:
                weights[i] = 0.1
                continue
            
            density_score = 0.0
            
            # لكل كلمة: حساب local density
            for j, word in enumerate(words):
                start = max(0, j - self.window_size)
                end = min(len(words), j + self.window_size + 1)
                context_words = words[start:end]
                
                # المعادلة: ?_local = |context| × (unique_ratio)
                unique_ratio = len(set(context_words)) / len(context_words)
                local_density = len(context_words) * unique_ratio
                
                # المعادلة: contribution = 1 + ln(1 + ?_local)
                contribution = 1 + np.log(1 + local_density)
                density_score += contribution
            
            # المعادلة النهائية: SDW = (1/|words|) ? contributions
            weights[i] = density_score / len(words)
            
            # ?? تسجيل
            eq_tracker.record_sdw(weights[i])
        
        return weights
    
    def extract_weighted_content(self, text: str, keep_ratio: float = 0.5) -> str:
        
        
        # تقسيم إلى chunks
        chunks = self.split_into_chunks(text)
        
        if len(chunks) <= 2:
            # نص قصير جداً - نأخذ أول chunk فقط
            return chunks[0] if len(chunks) > 0 else text
        
        # حساب أوزان الـ chunks
        weights = self.compute_chunk_weights(chunks)
        
        # اختيار أفضل X% من الـ chunks
        num_to_keep = max(1, int(len(chunks) * keep_ratio))
        
        # ترتيب حسب الوزن
        sorted_indices = sorted(weights.keys(), key=lambda x: weights[x], reverse=True)
        top_indices = sorted(sorted_indices[:num_to_keep])  # الحفاظ على الترتيب
        
        # دمج الـ chunks المختارة
        selected_chunks = [chunks[i] for i in top_indices]
        result = ' '.join(selected_chunks)
        
        # ? التأكد من أننا اختصرنا فعلاً
        original_words = len(text.split())
        extracted_words = len(result.split())
        
        if extracted_words >= original_words * 0.95:
            # لم نختصر بشكل كافٍ - نأخذ 40% بدلاً من 50%
            num_to_keep = max(1, int(len(chunks) * 0.4))
            top_indices = sorted(sorted_indices[:num_to_keep])
            selected_chunks = [chunks[i] for i in top_indices]
            result = ' '.join(selected_chunks)
        
        return result

# ============================================
# 7. المعادلة 3: Enhanced TF-IDF Keyword Extraction
# ============================================
class EnhancedKeywordExtractor:
    """
    ???????????????????????????????????????????????????????????????????
    المعادلة الرياضياتية 3: Enhanced TF-IDF with Multi-Factor Weighting
    ???????????????????????????????????????????????????????????????????
    
    الصيغة الرياضياتية:
    ????????????????????
    Score(w) = TF-IDF(w) × ?_topology(w) × W_position(w) × B_length(w)
    
    حيث:
    - TF-IDF(w) = (freq/|D|) × ln(|D| / freq)
    - ?_topology(w) = 1 + ln(1 + ?²_position / |D|)    (position variance)
    - W_position(w) = 1 / (1 + ln(1 + first_pos/|D|))  (early bonus)
    - B_length(w) = 1 + 0.5·e^(-0.5·((|w| - 8)/3)²)    (Gaussian bonus)
    
    الهدف: استخراج كلمات مفتاحية أكثر أهمية
    ???????????????????????????????????????????????????????????????????
    """
    
    def __init__(self):
        self.optimal_length = 8  # طول الكلمة الأمثل
    
    def extract_keywords(self, text: str, selected_sentences: List[str], n_keywords: int = 30) -> List[str]:
        """
        تطبيق المعادلة المحسّنة خطوة بخطوة
        """
        all_words = text.split()
        selected_words = ' '.join(selected_sentences).split()
        
        # الكلمات المرشحة (غير موجودة في الجمل المختارة)
        candidate_words = [w for w in all_words if w not in selected_words and 4 <= len(w) <= 15]
        
        if len(candidate_words) == 0:
            return []
        
        word_scores = {}
        total_words = len(all_words)  # |D| في المعادلة
        
        for word in set(candidate_words):
            freq = all_words.count(word)
            
            if 2 <= freq <= 30:  # كلمات متوسطة التكرار
                
                # ?????????????????????????????????????????????????????
                # المكون 1: TF-IDF
                # ?????????????????????????????????????????????????????
                # المعادلة: TF-IDF = TF × IDF
                tf = freq / total_words
                #    ?????   ???????????
                #    freq    |D|
                
                idf = np.log(total_words / (freq + 1))
                #     ???????????????????????????
                #           ln(|D|/freq)
                
                tf_idf = tf * idf
                #        ?????  ???
                #         TF   IDF
                
                # ?????????????????????????????????????????????????????
                # المكون 2: Position Weight (early words get bonus)
                # ?????????????????????????????????????????????????????
                # المعادلة: W_pos = 1 / (1 + ln(1 + pos/|D|))
                first_pos = all_words.index(word)
                pos_weight = 1.0 / (1 + np.log(1 + first_pos / total_words))
                #            ???????????????????????????????????????
                #                    1/(1 + ln(1 + pos/|D|))
                
                # ?? تسجيل
                eq_tracker.record_positional(pos_weight)
                
                # ?????????????????????????????????????????????????????
                # المكون 3: Length Bonus (Gaussian around optimal=8)
                # ?????????????????????????????????????????????????????
                # المعادلة: B_len = 1 + 0.5·e^(-0.5·((L - ?)/?)²)
                length_diff = (len(word) - self.optimal_length) / 3
                #             ???????????   ???????????????????   ???
                #                 L              ?                 ?
                
                gaussian_term = np.exp(-0.5 * length_diff**2)
                #               ???????????????????????????????
                #                      e^(-0.5·x²)
                
                length_bonus = 1.0 + 0.5 * gaussian_term
                #              ???????????????????
                #                 1 + 0.5·gaussian
                
                # ?????????????????????????????????????????????????????
                # المكون 4: Topological ? (position variance)
                # ?????????????????????????????????????????????????????
                # المعادلة: ? = 1 + ln(1 + ?²_pos/|D|)
                positions = [i for i, w in enumerate(all_words) if w == word]
                if len(positions) > 1:
                    pos_var = np.var(positions)
                    #         ?????????????
                    #            ?²_pos
                    
                    beta_local = 1 + np.log(1 + pos_var / total_words)
                    #            ???????????????????????????????
                    #                 1 + ln(1 + ?²/|D|)
                else:
                    beta_local = 1.0
                
                # ?????????????????????????????????????????????????????
                # المعادلة النهائية: دمج كل المكونات
                # ?????????????????????????????????????????????????????
                # Score = TF-IDF × ? × W_pos × B_len
                score = tf_idf * beta_local * pos_weight * length_bonus
                #       ???????   ???????????   ???????????   ?????????????
                #       TF-IDF      ?_topo      W_pos        B_len
                
                # ?? تسجيل
                eq_tracker.record_tfidf_enhanced(score)
                
                word_scores[word] = score
        
        # ترتيب حسب النتيجة
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [w for w, _ in sorted_words[:n_keywords]]

# ============================================
# 8. النظام الكامل
# ============================================
class ProvenMathematicalSystem:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        print(f"جاري التحميل: {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.dqs = DocumentQualityScorer(self.embedding_model)
        self.sdw = SemanticDensityWeighter(self.embedding_model)
        self.keyword_extractor = EnhancedKeywordExtractor()
        print("? النظام جاهز مع المعادلات الموثقة")
    
    def process_corpus(self, texts: List[str], labels: List[int]) -> Tuple:
        # المعادلة 1: DQS Filtering
        filtered_texts, filtered_labels = self.dqs.filter_corpus(texts, labels, keep_percentile=75)
        
        # المعادلة 2 + 3: SDW + Keywords
        print("\n?? تطبيق المعادلة 2: Semantic Density Weighting...")
        print("?? تطبيق المعادلة 3: Enhanced TF-IDF Keywords...")
        
        explanations = []
        extraction_stats = {
            'original_lengths': [],
            'extracted_lengths': [],
            'reduction_ratios': []
        }
        
        for text in tqdm(filtered_texts, desc="Extracting Explanations"):
            original_len = len(text.split())
            
            # SDW - استخراج أفضل chunks (50% من النص)
            ex = self.sdw.extract_weighted_content(text, keep_ratio=0.75)
            
            # Keywords - إضافة كلمات مفتاحية
            # نستخدم chunks بدلاً من sentences
            chunks = self.sdw.split_into_chunks(ex)
            keywords = self.keyword_extractor.extract_keywords(text, chunks, n_keywords=40)
            if keywords:
                ex = ex + ' ' + ' '.join(keywords)
            
            explanations.append(ex)
            
            # ?? تسجيل الإحصائيات
            extracted_len = len(ex.split())
            extraction_stats['original_lengths'].append(original_len)
            extraction_stats['extracted_lengths'].append(extracted_len)
            extraction_stats['reduction_ratios'].append(extracted_len / original_len if original_len > 0 else 1.0)
        
        # ? طباعة إحصائيات الاستخراج
        print("\n" + "="*70)
        print("?? إحصائيات الاستخراج")
        print("="*70)
        print(f"عدد الوثائق: {len(explanations)}")
        print(f"\nمتوسط طول Original: {np.mean(extraction_stats['original_lengths']):.1f} كلمة")
        print(f"متوسط طول Explanation: {np.mean(extraction_stats['extracted_lengths']):.1f} كلمة")
        print(f"متوسط نسبة الاستخراج: {np.mean(extraction_stats['reduction_ratios'])*100:.1f}%")
        print(f"أقل نسبة استخراج: {np.min(extraction_stats['reduction_ratios'])*100:.1f}%")
        print(f"أعلى نسبة استخراج: {np.max(extraction_stats['reduction_ratios'])*100:.1f}%")
        
        # ? عرض أمثلة
        print("\n" + "="*70)
        print("??Examples of extraction")
        print("="*70)
        
        for i in range(min(3, len(filtered_texts))):
            print(f"\n?? مثال {i+1}:")
            print(f"Original ({len(filtered_texts[i].split())} كلمة):")
            print(f"  {filtered_texts[i][:150]}...")
            print(f"\nExplanation ({len(explanations[i].split())} كلمة):")
            print(f"  {explanations[i][:150]}...")
            print(f"  نسبة الاستخراج: {(len(explanations[i].split())/len(filtered_texts[i].split()))*100:.1f}%")
        
        print("\n" + "="*70)
        
        # ? التحقق من أن الاستخراج يعمل
        same_count = sum(1 for i in range(len(filtered_texts)) 
                        if filtered_texts[i] == explanations[i])
        
        if same_count > len(filtered_texts) * 0.5:
            print(f"?? تحذير: {same_count}/{len(filtered_texts)} وثيقة متطابقة!")
            print("?? الاستخراج لا يعمل بشكل صحيح!")
        else:
            print(f"? الاستخراج يعمل: {len(filtered_texts) - same_count}/{len(filtered_texts)} وثيقة مختلفة")
        
        # إنشاء embeddings
        print("\n?? إنشاء embeddings...")
        doc_emb = self.embedding_model.encode(filtered_texts, show_progress_bar=True, batch_size=32)
        exp_emb = self.embedding_model.encode(explanations, show_progress_bar=True, batch_size=32)
        
        return doc_emb, exp_emb, explanations, filtered_texts, filtered_labels

# ============================================
# 9. BERTopic المحسّن
# ============================================
class OptimizedBERTopic:
    def fit(self, embeddings: np.ndarray, documents: List[str]):
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        
        umap_model = umap.UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=15,
            min_samples=10,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        vectorizer = CountVectorizer(
            stop_words="english",
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),
            max_features=5000
        )
        
        model = BERTopic(
            embedding_model=None,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            verbose=False,
            calculate_probabilities=True,
            nr_topics=20
        )
        
        topics, _ = model.fit_transform(documents, embeddings)
        return model, topics

# ============================================
# 10. التقييم
# ============================================
class Evaluator:
    def coherence(self, model, docs: List[str], coh_type: str = 'c_v') -> float:
        try:
            topics = {k: v for k, v in model.get_topics().items() if k != -1}
            if len(topics) == 0:
                return 0.0
            
            topic_words = [[w for w, _ in model.get_topic(tid)[:10]] for tid in topics]
            texts_tok = [d.split() for d in docs]
            dictionary = Dictionary(texts_tok)
            
            cm = CoherenceModel(
                topics=topic_words,
                texts=texts_tok,
                dictionary=dictionary,
                coherence=coh_type
            )
            return cm.get_coherence()
        except:
            return 0.0
    
    def diversity(self, model, top_n: int = 10) -> float:
        try:
            topics = {k: v for k, v in model.get_topics().items() if k != -1}
            if len(topics) == 0:
                return 0.0
            
            all_w, uniq_w = [], set()
            for tid in topics:
                words = [w for w, _ in model.get_topic(tid)[:top_n]]
                all_w.extend(words)
                uniq_w.update(words)
            
            return len(uniq_w) / len(all_w) if len(all_w) > 0 else 0.0
        except:
            return 0.0

# ============================================
# 11. التجربة النهائية مع إثبات المعادلات
# ============================================
def run_proven_mathematical_experiment():
    print("="*70)
    print("?? التجربة مع إثبات تطبيق المعادلات الرياضياتية")
    print("="*70)
    print("\nالمعادلات المطبقة:")
    print("1. DQS(d) = 0.4·LD + 0.4·SC + 0.2·LP")
    print("2. SDW(s) = (1/|w|) ?[1 + ln(1 + ?_local)]")
    print("3. Score(w) = TF-IDF × ? × W_pos × B_len")
    print("="*70)
    
    # تحميل البيانات
    prep = OptimizedPreprocessor()
    texts, labels, names = prep.load_20newsgroup()
    
    # عينة
    sample_size = 6000
    texts = texts[:sample_size]
    labels = labels[:sample_size]
    print(f"\nالعينة: {len(texts)} وثيقة")
    
    # المعالجة
    system = ProvenMathematicalSystem()
    doc_emb, exp_emb, explanations, filtered_texts, filtered_labels = system.process_corpus(texts, labels)
    
    # طباعة تقرير المعادلات
    eq_tracker.print_report()
    
    # التقييم
    modeler = OptimizedBERTopic()
    evaluator = Evaluator()
    
    results = {}
    
    # التكوينات الأربعة
    configs = {
        'Only original docs': (doc_emb, filtered_texts),
        'Only explanations': (exp_emb, explanations),
        'Concatenated': (np.concatenate([doc_emb, exp_emb], axis=1), filtered_texts),
        'Averaged': ((doc_emb + exp_emb) / 2, filtered_texts)
    }
    
    print("\n" + "="*70)
    print("التقييم - 4 تكوينات")
    print("="*70)
    
    for name, (emb, docs) in configs.items():
        print(f"\n{'='*70}")
        print(f"? {name}")
        print(f"{'='*70}")
        
        emb = np.nan_to_num(emb, nan=0.0, posinf=1.0, neginf=-1.0)
        
        model, topics = modeler.fit(emb, docs)
        n_topics = len(set(topics)) - (1 if -1 in topics else 0)
        
        cv = evaluator.coherence(model, docs, 'c_v')
        cnpmi = evaluator.coherence(model, docs, 'c_npmi')
        div = evaluator.diversity(model)
        
        results[name] = {
            'CV': cv,
            'CNPMI': cnpmi,
            'Diversity': div,
            'Topics': n_topics
        }
        
        print(f"المواضيع: {n_topics}")
        print(f"? CV: {cv:.4f}")
        print(f"? CNPMI: {cnpmi:.4f}")
        print(f"? Diversity: {div:.4f}")
    
    # النتائج
    print("\n" + "="*70)
    print("?? النتائج النهائية")
    print("="*70)
    
    df = pd.DataFrame(results).T.round(4)
    print("\n" + df.to_string())
    
    # إثبات أن explanations مختلفة
    print("\n" + "="*70)
    print("?? إثبات أن Explanations مختلفة عن Original")
    print("="*70)
    
    # حساب كم وثيقة متطابقة
    identical_count = 0
    for i in range(len(filtered_texts)):
        if filtered_texts[i].strip() == explanations[i].strip():
            identical_count += 1
    
    print(f"\n?? وثائق متطابقة: {identical_count}/{len(filtered_texts)} ({identical_count/len(filtered_texts)*100:.1f}%)")
    
    if identical_count > len(filtered_texts) * 0.8:
        print("? المشكلة: معظم الوثائق متطابقة - الاستخراج لا يعمل!")
    elif identical_count > len(filtered_texts) * 0.3:
        print("?? تحذير: نسبة عالية من الوثائق المتطابقة")
    else:
        print("? الاستخراج يعمل بشكل جيد")
    
    # عرض أمثلة مختلفة
    print("\n?? أمثلة على الوثائق المختلفة:")
    different_examples = 0
    for i in range(len(filtered_texts)):
        if filtered_texts[i][:200] != explanations[i][:200] and different_examples < 3:
            different_examples += 1
            print(f"\n?? مثال {different_examples}:")
            print(f"Original ({len(filtered_texts[i].split())} كلمة):")
            print(f"{filtered_texts[i][:200]}...")
            print(f"\nExplanation ({len(explanations[i].split())} كلمة):")
            print(f"{explanations[i][:200]}...")
    
    if different_examples == 0:
        print("\n? لم نجد أمثلة مختلفة - كل الوثائق متطابقة!")
        print("?? هذا يفسر لماذا النتائج متطابقة")
    
    # المقارنة
    print("\n" + "="*70)
    print("مقارنة مع الورقة")
    print("="*70)
    
    paper_baseline = {'CV': 0.68, 'CNPMI': 0.17}
    
    best_cv = max(results.items(), key=lambda x: x[1]['CV'])
    best_cnpmi = max(results.items(), key=lambda x: x[1]['CNPMI'])
    
    print(f"\n?? الورقة: CV={paper_baseline['CV']:.4f}, CNPMI={paper_baseline['CNPMI']:.4f}")
    
    print(f"\n?? أفضل CV ({best_cv[0]}): {best_cv[1]['CV']:.4f}")
    improvement_cv = ((best_cv[1]['CV'] - paper_baseline['CV']) / paper_baseline['CV']) * 100
    print(f"   التحسن: {improvement_cv:+.2f}%")
    if best_cv[1]['CV'] >= 0.70:
        print(f"   ? تحقق الهدف!")
    
    print(f"\n?? أفضل CNPMI ({best_cnpmi[0]}): {best_cnpmi[1]['CNPMI']:.4f}")
    
    # تحليل
    print("\n" + "="*70)
    print("?? تحليل تأثير المعادلات")
    print("="*70)
    
    only_orig = results['Only original docs']['CV']
    only_expl = results['Only explanations']['CV']
    
    print(f"\nOnly original: {only_orig:.4f}")
    print(f"Only explanations: {only_expl:.4f}")
    
    if abs(only_orig - only_expl) < 0.0001:
        print("?? تحذير: النتائج متطابقة - قد تكون هناك مشكلة!")
    else:
        diff = ((only_expl - only_orig) / only_orig) * 100
        print(f"? الفرق: {diff:+.2f}%")
        if only_expl > only_orig:
            print("? الـ Explanations أفضل!")
        else:
            print("?? الـ Explanations أسوأ - قد نحتاج لتعديل المعادلات")
    
    return df, model

# ============================================
# 12. التشغيل
# ============================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("?? بدء التجربة مع توثيق كامل للمعادلات...")
    print("="*70)
    
    results_df, final_model = run_proven_mathematical_experiment()
    
    print("\n" + "="*70)
    print("? التجربة اكتملت!")
    print("="*70)
    
    print("\n" + "="*70)
    print("?? ملخص المعادلات المطبقة")
    print("="*70)
    print("""
    المعادلة 1: Document Quality Scoring
    
    DQS(d) = 0.4·LD(d) + 0.4·SC(d) + 0.2·LP(d)
    
    السطر البرمجي:
    dqs = alpha * lexical_div + beta * semantic_coh + gamma * length_pen
    
    
    المعادلة 2: Semantic Density Weighting
    ????????????????????????????????????
    SDW(s) = (1/|words|) ?[1 + ln(1 + ?_local(w))]
    
    السطر البرمجي:
    contribution = 1 + np.log(1 + local_density)
    weights[i] = density_score / len(words)
    
    ????????????????????????????????????
    المعادلة 3: Enhanced TF-IDF
    ????????????????????????????????????
    Score(w) = TF-IDF × ?_topology × W_position × B_length
    
    السطر البرمجي:
    score = tf_idf * beta_local * pos_weight * length_bonus
    
    ????????????????????????????????????
    """)
    
    print("\n" + "="*70)
    print("?? أصل المعادلات:")
    print("="*70)
    print("""
    
    print("="*70)
