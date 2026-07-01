import matplotlib.pyplot as plt
import math
from time import time
from util import dict_to_filename
import pickle
import os
import json

def plot_top_words(model, feature_names, n_top_words, title):
    grid_side = int(math.ceil(math.sqrt(len(model.components_))))
    
    fig, axes = plt.subplots(grid_side, grid_side, figsize=(grid_side*5, grid_side*5), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 30})
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    # plt.show()
    return fig, axes

def fit_topic_model(train_docs_vect, vectorizer, model_class, hp_static, hp_dyn, prefix, analysis_path, overwrite_if_exists=False):
    model_desc = prefix + '_' + dict_to_filename(hp_dyn)
    if os.path.isdir(f'{analysis_path}/topic_models/{model_desc}/model.pkl') and not overwrite_if_exists:
        print(f'found a file at {f'{analysis_path}/topic_models/{model_desc}/model.pkl'}, skipping')
        return
    t0 = time()
    print(f'fitting model: {model_desc}...')
    model = model_class(**hp_static, **hp_dyn).fit(train_docs_vect)
    print(f'done in {time()-t0}s')
    fig, _ = plot_top_words(model, vectorizer.get_feature_names_out(), n_top_words=15, title=model_desc)
    os.makedirs(f'{analysis_path}/topic_models/{model_desc}', exist_ok=True)
    with open(f'{analysis_path}/topic_models/{model_desc}/model.pkl', 'wb') as f: 
        pickle.dump(model, f)
    with open(f'{analysis_path}/topic_models/{model_desc}/vectorizer.pkl', 'wb') as f: 
        pickle.dump(vectorizer, f)
    fig.savefig(f'{analysis_path}/topic_models/{model_desc}/topics.png')
    plt.close()
    with open(f'{analysis_path}/topic_models/{model_desc}/model.hyperparams.json', 'w') as f: 
        json.dump(hp_dyn | hp_static, f)

def load(topic_model):
    with open(f'{topic_model}/model.pkl', 'rb') as f: 
        model = pickle.load(f)
    with open(f'{topic_model}/vectorizer.pkl', 'rb') as f: 
        vectorizer = pickle.load(f)
    return model, vectorizer