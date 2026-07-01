from collections import Counter
import json
import math
import numpy as np
from typing import Tuple, Dict, Optional
from scipy import sparse
import re, string
pattern = re.compile('[^a-zA-Z0-9_ ]+')
from glob import glob
import pickle
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
import itertools
import os
from functools import lru_cache
# from picky_bpe.bpe_trainer import BPE as picky
# from tokenizers import trainers
# from tokenizers import Tokenizer
# from tokenizers.models import BPE, Unigram
# from tokenizers.pre_tokenizers import Whitespace, PreTokenizer, ByteLevel, Sequence, CharDelimiterSplit
# from tokenizers.normalizers import Lowercase

strip_punct = lambda s: pattern.sub('', s)
text_to_tokens = lambda text: strip_punct(text.lower()).strip().split()
def read_docs(docfiles, max_size=None): 
    docs = []
    size_so_far = 0
    for fpath in docfiles: 
        with open(fpath) as f: 
            docs.append(' '.join(text_to_tokens(f.read())))
        size_so_far += len(docs[-1])
        if max_size and size_so_far >= max_size:
            break
    return docs 

def to_camel_case(s: str) -> str:
    """Convert string (with spaces, underscores, hyphens) to camelCase."""
    parts = re.split(r'[\s_-]+', s.strip())
    if not parts:
        return ""
    return parts[0].lower() + "".join(word.capitalize() for word in parts[1:])

def safe_filename(name: str) -> str:
    """Replace unsafe filename characters with underscores."""
    return re.sub(r'[^A-Za-z0-9._-]', '_', name)

def dict_to_filename(data: dict) -> str:
    """Embed dict into a safe filename with camelCase keys and string values camelCased."""
    parts = []
    for k, v in data.items():
        key = to_camel_case(str(k))
        if isinstance(v, str):
            value = to_camel_case(v)
        else:
            value = str(v)
        parts.append(f"{key}-{value}")
    filename = "_".join(parts)
    return safe_filename(filename)

def list_dataset_files(dataset, lang="en", corpora_root="/Users/rastislavhronsky/research/corpora"): 
    return [fn for fn in glob(os.path.join(corpora_root, dataset, lang, '**'), recursive=True) if fn.endswith('.txt')]

# ---- metrics ----

def kld_from_uniform(doc_term, columnwise=True): 
    relative = (doc_term + 1e-16) / (doc_term.sum(axis=0 if columnwise else None) + 1e-16)
    shannon = -(relative * np.log2(relative)).sum(axis=0 if columnwise else None)
    shannon_uniform = np.log2(doc_term.shape[0] if columnwise else np.prod(doc_term.shape))
    dispersion = shannon / shannon_uniform
    return dispersion

def dispersion_across_docs(doc_term): 
    """for each term, quantifies how evenly it occurs in documents
    """
    return kld_from_uniform(doc_term)

def dispersion_across_terms(doc_term): 
    """for every document, quantifies how evenly the terms are distributed
    """
    return kld_from_uniform(doc_term.T)

def dispersion_overall(doc_term): 
    """dispersion of the whole document-term matrix
    """
    return kld_from_uniform(doc_term, columnwise=False)

def shannon_efficiency(doc_term): 
    """for reference, shannon efficiency"""
    return kld_from_uniform(doc_term.T.sum(axis=1).reshape(-1, 1))[0]

def average_js_to_mean_sparse_csr(
    X: sparse.csr_matrix,
    mean_thresh: float = 1e-8,
    tail_frac: float = 0.01,
    bootstrap: Optional[Dict] = None,
    eps: float = 1e-12
) -> Tuple[float, float, np.ndarray, Dict]:
    """
    Compute average Jensen–Shannon divergence between rows of a CSR sparse matrix
    (each row = categorical distribution of counts) and their mean, 
    grouping the trailing `tail_frac` of mean probability mass into one tail bucket.

    Parameters
    ----------
    X : csr_matrix, shape (n_samples, n_categories)
        Each row contains counts for a categorical distribution.
    mean_thresh : float, optional
        Ignore categories with mean < mean_thresh before applying tail fraction.
    tail_frac : float, optional
        Fraction of mean probability mass to merge into the tail (default 0.01 = last 1%).
    bootstrap : dict, optional
        {'n_boot':200, 'alpha':0.05, 'seed':0} for bootstrap CI.
    eps : float, optional
        Additive smoothing constant for numerical stability.

    Returns
    -------
    avg_js_nats : float
        Average JS divergence (in nats).
    avg_js_bits : float
        Average JS divergence (in bits).
    per_js_bits : np.ndarray
        Per-observation JS divergences (in bits).
    meta : dict
        Contains kept indices, mean vector, tail mass, bootstrap info if requested.
    """
    assert sparse.isspmatrix_csr(X), "Input must be CSR matrix"
    n, K = X.shape
    if n == 0 or K == 0:
        return 0.0, 0.0, np.array([]), {}

    # Normalize rows to probability distributions
    row_sums = np.array(X.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    P = X.multiply(1.0 / row_sums[:, None])

    # Compute mean probability per category
    mean = np.array(P.mean(axis=0)).ravel()

    # Filter out near-zero mean categories first
    valid_mask = mean >= mean_thresh
    mean = mean[valid_mask]
    kept_indices_all = np.nonzero(valid_mask)[0]

    # Sort by descending mean
    order = np.argsort(mean)[::-1]
    sorted_mean = mean[order]

    # Keep just enough categories to cover (1 - tail_frac) of total mass
    cumsum = np.cumsum(sorted_mean)
    cutoff_idx = np.searchsorted(cumsum, (1 - tail_frac) * cumsum[-1], side="right")
    cutoff_idx = min(cutoff_idx, len(sorted_mean))
    kept_idx_local = order[:cutoff_idx]
    kept_idx = kept_indices_all[kept_idx_local]
    mean_kept = mean[kept_idx_local]
    tail_mean = 1.0 - mean_kept.sum()

    # Compute per-observation JS divergences
    per_js = np.zeros(n)
    q_vec = np.append(mean_kept, tail_mean)
    for i in range(n):
        start, end = X.indptr[i], X.indptr[i + 1]
        cols = X.indices[start:end]
        vals = P.data[start:end]
        p_vec = np.zeros(len(kept_idx) + 1)
        mask = np.isin(cols, kept_idx)
        if np.any(mask):
            # Map kept indices to local positions
            idx_map = {cat: j for j, cat in enumerate(kept_idx)}
            for c, v in zip(cols[mask], vals[mask]):
                p_vec[idx_map[c]] = v
        p_vec[-1] = max(0.0, 1.0 - p_vec[:-1].sum())

        # JS divergence with smoothing
        p = np.maximum(p_vec, eps)
        q = np.maximum(q_vec, eps)
        p /= p.sum(); q /= q.sum()
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log(p / m))
        kl_qm = np.sum(q * np.log(q / m))
        per_js[i] = 0.5 * (kl_pm + kl_qm)

    avg_js = float(per_js.mean())
    avg_js_bits = avg_js / math.log(2)
    per_js_bits = per_js / math.log(2)

    meta = {
        'kept_idx': kept_idx,
        'mean_kept': mean_kept,
        'tail_mean': tail_mean,
        'tail_frac': tail_frac,
        'K': K
    }

    # Optional bootstrap CI
    if bootstrap:
        n_boot = int(bootstrap.get('n_boot', 200))
        alpha = float(bootstrap.get('alpha', 0.05))
        seed = bootstrap.get('seed', None)
        rng = np.random.default_rng(seed)
        boot_means = [per_js[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
        ci_nats = (np.quantile(boot_means, alpha / 2),
                   np.quantile(boot_means, 1 - alpha / 2))
        meta['bootstrap'] = {
            'n_boot': n_boot, 'alpha': alpha,
            'ci_nats': ci_nats,
            'ci_bits': (ci_nats[0] / math.log(2), ci_nats[1] / math.log(2))
        }

    return avg_js, avg_js_bits, per_js_bits, meta

# ---- vocabulary building utilities ----

def halving_hyperplanes(points, depth, seed=None):
    """
    Recursively split points into 2^depth subsets using halving hyperplanes.

    Args:
        points: numpy array (n, d)
        depth: how many recursive splits (number of regions = 2^depth)
        seed: random seed for reproducibility

    Returns:
        regions: list of point indices for each region
        hyperplanes: list of (w, b) defining hyperplanes used
    """
    rng = np.random.default_rng(seed)
    n, d = points.shape
    hyperplanes = []
    regions = []

    def recurse(idx_list, current_depth):
        # Base case: stop splitting
        if current_depth == depth:
            regions.append(idx_list)
            return

        # Pick random direction
        w = rng.normal(size=d)
        w = w / np.linalg.norm(w)

        # Project points
        projections = points[idx_list] @ w
        b = np.median(projections)

        hyperplanes.append((w, b))

        # Partition indices
        left_idx = [i for i in idx_list if points[i] @ w <= b]
        right_idx = [i for i in idx_list if points[i] @ w > b]

        # Recurse
        recurse(left_idx, current_depth + 1)
        recurse(right_idx, current_depth + 1)

    recurse(list(range(n)), 0)
    return regions, hyperplanes

alphabet = list(map(chr, [9,10,13] + list(range(32,127))))
alphabet_set = set(alphabet)

class VocabFilter: 
    tf: Counter
    df: Counter

    def __init__(self, term_freq, doc_freq={}):
        self.tf = term_freq
        assert self.tf, 'term frequency has to be populated'
        self.df = doc_freq

    @lru_cache(maxsize=8)
    def filter(self, min_df=1, min_tf=1, order_by='df', max_size=None):
        ref_counter = self.df if order_by=='df' and self.df else self.tf
        ref_terms, ref_stat = zip(*ref_counter.most_common())
        terms_and_stats = [(term, stat) for term, stat in zip(ref_terms, ref_stat) if self._passes(term, min_df, min_tf)]
        if isinstance(max_size, int) and max_size > 0: 
            terms_and_stats = terms_and_stats[:max_size]
        terms, stats = zip(*terms_and_stats)

        missing = alphabet_set.difference(terms)
        if missing: terms = terms[:-len(missing)]
        # return the final set of terms and the last passing stat (criterion, e.g. document frequency)
        return set(terms).union(missing), stats[-1-len(missing)]
    
    def _passes(self, term, min_df, min_tf): 
        df_ok = term not in self.df or self.df[term] >= min_df
        tf_ok = term not in self.tf or self.tf[term] >= min_tf
        return df_ok and tf_ok
    def __len__(self): 
        return len(self.tf)
    def intersection(self, other): 
        return set(self.tf).intersection(other.tf)
    def union(self, other): 
        return set(self.tf).union(other.tf)
    def difference(self, other): 
        return set(self.tf).difference(other.tf)

def _check_files(file_or_files):
    if isinstance(file_or_files, (list, tuple, set)): 
        files = file_or_files
    elif isinstance(file_or_files, str): 
        files = [file_or_files]
    else: 
        raise ValueError(f'received type {type(file_or_files)} for file_or_files, dont know what to do with it :(')
    return files

# def build_bpe_vocab(file_or_files, drop_unused=False,  normalizer=None, pre_tokenizer=None, max_token_length=20, vocab_size=1000000) -> Counter: 
#     # global bpe_trainer, pre_tokenizer, normalizer, alphabet
#     files = _check_files(file_or_files)
#     tokenizer = Tokenizer(BPE())
#     tokenizer.pre_tokenizer = pre_tokenizer
#     tokenizer.normalizer = normalizer
#     bpe_trainer = trainers.BpeTrainer(vocab_size=vocab_size, 
#                                         show_progress=False, 
#                                         initial_alphabet=alphabet, 
#                                         min_frequency=2,
#                                         max_token_length=max_token_length)
#     tokenizer.train(files, bpe_trainer)
#     if drop_unused: 
#         # load and encode files 
#         file_contents = [open(f).read() for f in files]
#         decoded_tokens = [tok for enc in tokenizer.encode_batch(file_contents) for tok in enc.tokens]
#         res_without_alphabet = Counter(decoded_tokens)
#     else:
#         merges = json.loads(tokenizer.to_str())["model"]["merges"]
#         res_without_alphabet = Counter({s1+s2: freq for s1, s2, freq in merges})
#     # add 1 to every token count as well (the alphabet-counter below will also add a finctional 1 to every letter)
#     res_without_alphabet = Counter({word: count+1 for word, count in res_without_alphabet.most_common()})
#     res = res_without_alphabet + Counter(alphabet)
#     return res

# def build_picky_bpe_vocab(file_or_files, threshold=.6, coverage=1.0, vocab_size=100000, normalizer=None, pre_tokenizer=None) -> Counter: 
#     files = _check_files(file_or_files)
#     contents = []
#     for file in files: 
#         if os.path.isfile(file): 
#             with open(file) as f: 
#                 content = f.read()
#         else: 
#             content = file
#         if normalizer: 
#             content = normalizer.normalize_str(content)
#         if len(content) == 0: 
#             print(f'Warning: file {file} was empty after normalizing')
#             continue
#         if pre_tokenizer: 
#             pretokens, _ = zip(*pre_tokenizer.pre_tokenize_str(content))
#             content = ' '.join(pretokens) # picky_bpe will split by whitespace
#         contents.append(content)
#     trainer = picky(vocab_size=vocab_size, threshold=threshold, coverage=coverage)
#     tokens, freqs = trainer.fit_return('\n'.join(contents))
#     tokens = list(map(lambda s: s.replace('▁', ' '), tokens))
#     res_without_alphabet = {token: freq + 1 for token, freq in zip(tokens, freqs)}
#     res_without_alphabet = Counter(res_without_alphabet)
#     res = res_without_alphabet + Counter(alphabet)
#     return res

# def build_unigram_vocab(file_or_files, normalizer=None, pre_tokenizer=None, vocab_size=100000, max_token_length=20) -> Counter: 
#     files = _check_files(file_or_files)
#     tokenizer = Tokenizer(Unigram())
#     tokenizer.pre_tokenizer = pre_tokenizer
#     tokenizer.normalizer = normalizer
#     trainer = trainers.UnigramTrainer(vocab_size=vocab_size, 
#                                         show_progress=False, 
#                                         initial_alphabet=alphabet, 
#                                         max_piece_length=max_token_length)
#     tokenizer.train(files, trainer)
#     # load and encode files to get token frequencies
#     file_contents = [open(f).read() for f in files]
#     decoded_tokens = [tok for enc in tokenizer.encode_batch(file_contents) for tok in enc.tokens]
#     res = Counter(decoded_tokens)
#     for char in alphabet: 
#         if char not in res: 
#             res[char] = 1
#     return res

# def build_vocab_with_doc_stats(files, vocab_builder, min_df=1):
#     doc_freq = Counter()
#     term_freq = Counter()
#     for file in files: 
#         try: 
#             vocab = vocab_builder(file)
#         except Exception as e: 
#             print(f"Warning: vocabulary builder failed for file {file} (error message: {str(e)})")
#             continue
#         term_freq.update(vocab)
#         types, _ = zip(*vocab.items())
#         doc_freq.update(types)
#     return (term_freq, doc_freq)

# def renyi_efficiency(counts, alpha):
#     p = counts / counts.sum()
#     n = len(p)

#     if alpha == 1.0:
#         H = -np.sum(np.where(p > 0, p * np.log2(p), 0.0))
#     else:
#         H = (1 / (1 - alpha)) * np.log2(np.sum(p ** alpha))
#     return H / np.log2(n)

# def terms_to_tokenizer(terms, pre_tokenizer, normalizer): 
#     tokenizer = Tokenizer(BPE())
#     tokenizer.add_tokens(terms)
#     tokenizer.pre_tokenizer = pre_tokenizer
#     tokenizer.normalizer = normalizer
#     return tokenizer

# @lru_cache(maxsize=10000)
# def read_file(file): 
#     if os.path.isfile(file): 
#         with open(file) as f: 
#             raw_content = f.read()
#     else: 
#         raw_content = file 
#     content = ''.join(ch if ch in alphabet_set else '_' for ch in raw_content)
#     return content

# def eval_vocab(files_or_docs, vocab_terms, normalizer, pre_tokenizer, tokenized_out=None, show_progressbar=False): 
#     if isinstance(vocab_terms, set): 
#         vocab_terms = sorted(vocab_terms)
#     tokenizer = terms_to_tokenizer(vocab_terms, pre_tokenizer=pre_tokenizer, normalizer=normalizer)
#     stoi = dict(zip(vocab_terms, range(len(vocab_terms))))
#     total_content_size = total_token_count = num_whs = 0
#     counts = np.zeros((len(files_or_docs), len(stoi)), dtype=np.int32) # will contain the document-term count matrix after the for loop
#     tokenization_sample = ''
#     for file_i, file in tqdm(enumerate(files_or_docs), total=len(files_or_docs), disable=not show_progressbar): 
#         content = read_file(file)
#         content_size = len(content.encode())
#         total_content_size += content_size
#         tokens = tokenizer.encode(content).tokens
#         num_whs += sum(token==' ' for token in tokens)

#         if len(tokenization_sample) < 5000: 
#             tokens_sample = tokenizer.encode(content[:100]).tokens
#             tokens_sample = [t.replace('_', '???').replace(' ', '_') for t in tokens_sample]
#             tokenization_sample += ' '.join(tokens_sample) + '\n'

#         tokenization_ttl_len = sum(map(len, tokens))
#         tokens_missing = content_size - tokenization_ttl_len
#         total_token_count += tokens_missing

#         if tokenization_ttl_len != content_size: 
#             print(f'Warning: length of tokenization output ({tokenization_ttl_len}) does not match length of raw content ({len(content)}); file: {file}')
#         if isinstance(tokenized_out, list):
#             tokenized_out.extend(tokens)
#         for term, count in Counter(tokens).items(): 
#             counts[file_i, stoi[term]] = count
#             total_token_count += count
    
#     token_length_distribution = [0 for _ in range(max(map(len, vocab_terms)))]
#     for term in stoi: 
#         token_length_distribution[len(term)-1] += int(counts[:, stoi[term]].sum())

#     # avg_js, avg_js_bits, per_js_bits, meta = average_js_to_mean_sparse_csr(counts, **jsd_kwargs)
#     # entropy_bits, renyi_efficiency, renyi_entropy_bits = compute_entropy_and_renyi_efficiency(
#     #     meta['mean_kept'], meta['tail_mean'], base=2, alpha=renyi_efficiency_alpha
#     # )

#     result = dict(
#         bytes_per_token=total_content_size/total_token_count,
#         total_bytes=total_content_size,
#         total_tokens=total_token_count,
#         total_whitespace=num_whs,
#         dispersion_across_terms=dispersion_across_terms(counts).mean().item(),
#         dispersion_across_docs=dispersion_across_docs(counts).mean().item(),
#         dispersion_overall=dispersion_overall(counts).item(),
#         token_length_distribution=token_length_distribution,
#         sample=tokenization_sample,
#         vocab_size=len(vocab_terms),
#         freq=counts.sum(axis=0),
#     )
#     counts_ttl = counts.sum(axis=0)
#     for alpha in [1.0, 1.5, 2.0, 2.5, 3.0]: 
#         result[f'eff_renyi_{alpha:.1f}'] = float(renyi_efficiency(counts_ttl, alpha))
#     return result

# def partition_dataset(files, topic_model_path, n_regions=14, random_state=1, batch_size=1000):
#     with open(f'{topic_model_path}/model.pkl', 'rb') as f: 
#         model = pickle.load(f) 
#     with open(f'{topic_model_path}/vectorizer.pkl', 'rb') as f: 
#         vectorizer = pickle.load(f)

#     docs_latent_list = []
#     sizes = []
#     for filebatch in tqdm(itertools.batched(files, batch_size), total=len(files)//batch_size, desc='reading text batches...'):
#         batch = read_docs(filebatch)
#         sizes += [len(doc.encode()) for doc in batch]
#         _docs_tfidf = vectorizer.transform(batch)
#         _docs_latent = model.transform(_docs_tfidf)
#         docs_latent_list.append(_docs_latent)
#     docs_latent = np.vstack(docs_latent_list)

#     kmeans = KMeans(n_clusters=n_regions, random_state=random_state)
#     assignments = kmeans.fit_predict(docs_latent)
    
#     sizes_region = [0 for r in range(n_regions)]
#     for doc_id, asgn in enumerate(assignments): 
#         sizes_region[asgn] += sizes[doc_id]

#     latent_part = [docs_latent[assignments==r].mean(axis=0) for r in range(n_regions)]
#     pairwise_dist = {
#         'cosine': pairwise_distances(np.array(latent_part), metric='cosine'),
#         'euclid': pairwise_distances(np.array(latent_part), metric='euclidean'),
#     }
#     files_part = [[files[doc_id] for doc_id in idx] for idx in regions]
#     return files_part, latent_part, pairwise_dist, sizes_region