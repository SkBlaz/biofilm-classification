import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pandas import Series


def parse_file_to_df(fname):


    ddl = defaultdict(list)
    with open(fname, encoding="cp1251") as inpfn:
        names = []
        distances = []
        percents = []
        positions = []
        full_names = []
        for line in inpfn:
            if "Image name:" in line:
                nparts = line.split("\t")[1:][0].split(" ")[2].split("_")
                aname = nparts[0] + "_" + nparts[6]
                pos = nparts[7]
                names.append(aname)
                positions.append(pos)
                full_names.append("-".join(nparts))
            
            if "Distance (" in line:
                distancesx = [
                    float(x.replace(",", "."))
                    for x in line.split("\t")[2:]
                ]
                distances.append(distancesx)
                
            elif "Percent:" in line:
                percx = [
                    float(x.replace(",", "."))
                    for x in line.split("\t")[2:]
                ]
                percents.append(percx)

    assert len(distances) == len(names) == len(percents)

    for enx in range(len(names)):
        ddl[names[enx]].append({"Distance": distances[enx], "Percent": percents[enx], "Positions": positions[enx], "full_name": full_names[enx]})
    return ddl


def visualize_basic_distributions(dist_storage):

    for k, v in dist_storage.items():
        for measurement in v:
            distances = measurement['Distance']
            percentages = measurement['Percent']
            plt.scatter(distances, percentages, label=measurement['Positions'])
        plt.legend()
        print(f"output_images/{k}.pdf")
        plt.xlabel("Distance (µm)")
        plt.ylabel("Percent")
        plt.title(k)
        plt.tight_layout()
        plt.savefig(f"output_images/{k}.pdf", dpi=300)
        plt.clf()
        plt.cla()


def visualize_interactive(dist_storage):

    # Create the scatter plot (for the moment: a blank graph)
    fig = go.Figure(layout=go.Layout(scattermode='group'))

    dates = []

    colors = ['red', 'blue', 'green']

    for k, v in dist_storage.items():
        
        print(k)
        d, label_value = k.split('_')

        if d not in dates:
            dates.append(d)

        color_idx = dates.index(d)
        
        # Add the scatter trace with color based on the category_variable
        #for category_data in v:
        category_data = v
        fig.add_trace(go.Scatter(
            x=category_data['Distance'],
            y=category_data['Average'],
            mode='markers',
            name=f"{d} {label_value}",
            error_y=go.scatter.ErrorY(array=category_data['Max'], arrayminus=category_data['Min']),
            marker=dict(
                size=12,
                opacity=0.7,
                color=colors[color_idx],
                line=dict(width=2, color='black') # Properties of the edges
            )
        ))
        
    fig.update_layout(
        title="Densities",
        xaxis_title="Distance (µm)",
        yaxis_title="Percent",
        width=1800,
        height=1200,
    )
        

    fig.write_html("output_images/interactive_avg.html")


def average_and_errors(dist_storage):
    ret = {}
    for k, v in dist_storage.items():
        suma = Series([0] * 21)
        max_val = Series([-1] * 21)
        min_val = Series([101] * 21)
        for pos in v:
            s_pos = Series(pos['Percent'])
            suma += s_pos
            max_val = max_val.combine(s_pos, max, 0)
            min_val = min_val.combine(s_pos, min, 0)
        suma /= len(v)
        ret[k] = {
            'Distance': [0.5 * i for i in range(0, 21)],
            'Average': suma,
            'Max': max_val - suma,
            'Min': suma - min_val
        }
    return ret

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Visualization of densities ..',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        '--input_txt_file',
        type=str,
        default='./data/L1323_AreaOcc.txt',
        help='Inpiut txt file.',
    )

    args = parser.parse_args()

    img_storage = parse_file_to_df(args.input_txt_file)
#    visualize_basic_distributions(img_storage)
    visualize_interactive(average_and_errors(img_storage))
