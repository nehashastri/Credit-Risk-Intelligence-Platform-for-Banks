import plotly.express as px

def line_chart(df, x, y, title):
    fig = px.line(df, x=x, y=y, title=title)
    fig.update_layout(height=400, margin=dict(l=20,r=20,t=40,b=20))
    return fig
