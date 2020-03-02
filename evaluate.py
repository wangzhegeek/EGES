
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_embeddings(embebed_mat, brand_list, shop_list, cate_list):
    model = TSNE(n_components=2)
    node_pos = model.fit_transform(embebed_mat)
    brand_color_idx, shop_color_idx, cate_color_idx = {}, {}, {}
    for i in range(len(node_pos)):
        brand_color_idx.setdefault(brand_list[i], [])
        brand_color_idx[brand_list[i]].append(i)
        shop_color_idx.setdefault(shop_list[i], [])
        shop_color_idx[shop_list[i]].append(i)
        cate_color_idx.setdefault(cate_list[i], [])
        cate_color_idx[cate_list[i]].append(i)

    plt.figure()
    for c, idx in brand_color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)  # c=node_colors)
    plt.title('brand distribution')
    plt.savefig('./data_cache/brand_dist.png')

    plt.figure()
    for c, idx in shop_color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)  # c=node_colors)
    plt.title('shop distribution')
    plt.savefig('./data_cache/shop_dist.png')

    plt.figure()
    for c, idx in cate_color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)  # c=node_colors)
    plt.title('cate distribution')
    plt.savefig('./data_cache/cate_dist.png')
