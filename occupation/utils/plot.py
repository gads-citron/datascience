import plotly.graph_objects as go
import citron_colors as col

def plot_occ(raw, occ, title = ""):
    fig = go.Figure()
    # fig.add_trace(go.Scatter(name = "power", x=occ.index, y=raw,
    #                          mode="lines", line={'color': green, 'width': 1, 'dash':'dot'}))
    fig.add_trace(go.Scatter(name = "occupancy", x=occ.index, y=occ.value,
                             mode="lines", line={'color': col.blue, 'width': 1}))
    fig.update_layout(title=title,
                      xaxis_title="Time", yaxis_title="power / occupancy",
                      template="simple_white")
    fig.update_xaxes(ticks='outside',
                     dtick=str(24*60*60*1000),
                     tickangle=-45,
                     tickformat='%a-%b-%e')
    fig.show()