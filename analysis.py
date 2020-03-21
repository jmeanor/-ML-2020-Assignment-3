from matplotlib import pyplot as plt
import seaborn as sns
import os 

def heatmap(data, path, name, color):
    # xlabels = ['host_id','latitude','longitude','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365','room_type__Entire home/apt','room_type__Private room','room_type__Shared room']
    
    cor = data['data'].corr()
    cor.to_csv(os.path.join(path, name+"-correlations.csv"))

    # cor = cor.loc[:,xlabels]

    ax = sns.heatmap(cor,cmap=color, square = True, xticklabels=True, yticklabels=True)
    ax.tick_params(axis="y", direction="inout")
    ax.tick_params(axis="x",  labelrotation=90.0)

    # ax.set_xticks(range(1, len(xlabels)))
    # ax.set_xticklabels(range(1, len(xlabels)))

    for tick in ax.yaxis.get_major_ticks()[1::2]:
        tick.set_pad(15)
    plt.show()