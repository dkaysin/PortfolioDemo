import numpy as np
import scipy.stats as stats

from bokeh.layouts import row, column, widgetbox, layout
from bokeh.models import Slider, Paragraph, Div, LabelSet, Label, HoverTool
from bokeh.plotting import figure, ColumnDataSource, curdoc


def calcCAGR(first, last):
    res = (last / first) ** (1 / nYears) - 1
    return "{0:.1%}".format(res)


def labelCAGR(prefix):
    return prefix + " " + str(nYears) + "Y: "


class Forecaster:
    mean = None
    stdev = None
    corr = None
    s0 = None
    years = None
    var = None
    covar = None
    meanCont = None
    dur = None

    def __init__(self, mean, stdev, corr, s0, nyears, dur):
        self.mean = mean
        self.stdev = stdev
        self.corr = corr
        self.s0 = s0
        self.years = np.array([range(0, nyears)])

        self.var = stdev ** 2
        self.covar = stdev.T * corr * stdev
        self.meanCont = np.log(1 + mean)

        self.dur = dur

    def calc(self, w):

        alpha = 0.1

        # Portfolio return
        t = self.years
        varP = w.T @ self.covar @ w
        stdevP = np.sqrt(varP)

        # start of VC
        allocVC = w[4][0] * s0
        ticketVC = 2.0
        pSuccessVC = 0.2
        multVC = 10.0
        yearsVC = 3.0
        nVC = np.round(allocVC / ticketVC, 0)
        if nVC >= 1:
            deltaVC = 0 # allocVC - nVC * ticketVC
            percVC = stats.binom.interval(1 - alpha * 2, nVC, pSuccessVC)
            meanVC = (nVC * pSuccessVC * ticketVC * multVC + deltaVC * 1.04) / allocVC
            topVC = (percVC[0] * ticketVC * multVC + deltaVC * 1.04) / allocVC
            bottomVC = (percVC[1] * ticketVC * multVC + deltaVC * 1.04) / allocVC
        else:
            meanVC = 0
            topVC = 0
            bottomVC = 0
        if nVC == 1:
            meanVC = allocVC * pSuccessVC * multVC
        # self.meanCont[4][0] = meanVC ** (1/yearsVC) - 1 ###!!!
        topPerAsset = self.meanCont.copy()
        bottomPerAsset = self.meanCont.copy()
        topPerAsset[4][0] = topVC ** (1/yearsVC) - 1
        bottomPerAsset[4][0] = bottomVC ** (1/yearsVC) - 1
        topP = w.T @ topPerAsset
        bottomP = w.T @ bottomPerAsset

        # end of VC

        meanP = w.T @ self.meanCont
        #
        # print("*****")
        # print(topP)
        # print(meanP)
        # print(bottomP)


        # meanLogR = np.log(self.s0) + (meanP - varP / 2) * t
        meanLogR = np.log(self.s0) + (meanP) * t
        varLogR = varP * t
        stdevLogR = np.sqrt(varLogR)

        topR = np.exp(meanLogR + stdevLogR * stats.norm.ppf(1 - alpha))
        meanR = np.exp(meanLogR)
        bottomR = np.exp(meanLogR + stdevLogR * stats.norm.ppf(alpha))

        # Duration
        durR = self.dur @ w

        periods = np.array(
            [[1.0 / 365],
             [5.0 / 365],
             [1.0 / 12],
             [1],
             [3]]
        )

        compoundedM = (1 + mean.T) ** periods
        # compoundedR = list(map(lambda x: x[0] @ x[1], zip(compoundedM.T, dur.T)))
        durCompR = (dur * compoundedM) @ w

        return topR, meanR, bottomR, durR.T, durCompR.T


assets = [
    "Cash",
    "Fixed Income Bonds",
    "Public Equities",
    "Real Estate",
    "Venture Capital",
    "Crypto"
]

mean = np.array(
    [[0.0200],
     [0.0500],
     [0.1400],
     [0.0600],
     [0.2000],
     [0.1000]]
)
stdev = np.array(
    [[0.0117],
     [0.0400],
     [0.1500],
     [0.0800],
     [0.3000],
     [0.4000]]
)
corr = np.array(
    [[1.00, -0.01, 0.04, 0.01, 0.00, -0.50],
     [-0.01, 1.00, 0.47, 0.06, 0.30, -0.70],
     [0.04, 0.47, 1.00, 0.053, 0.50, -0.80],
     [0.01, 0.06, 0.53, 1.00, 0.30, -0.00],
     [0.00, 0.30, 0.50, 0.30, 1.00, -0.40],
     [-0.50, -0.70, -0.80, 0.00, -0.40, 1.0]]
)

dur = np.array(
    [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.5, 0.0, 0.0, 0.5],
     [0.0, 0.0, 0.5, 0.0, 0.0, 0.2],
     [0.0, 0.0, 0.0, 0.4, 0.0, 0.3],
     [0.0, 0.0, 0.0, 0.6, 1.0, 0.0]]
)

durNames = [
    "In a day",
    "In a week",
    "In a month",
    "In a year",
    "In 3 years"
]

l = len(assets)
w = np.repeat(np.array([[1 / l]]), l, axis=0)

s0 = 150
nYears = 5

forecast = Forecaster(mean, stdev, corr, s0, nYears + 1, dur)

topR, meanR, bottomR, durR, durCompR = forecast.calc(w)

sourcePlot = ColumnDataSource(data=dict(
    x=forecast.years[0],
    y1=np.round(topR[0], 0),
    y2=np.round(meanR[0], 0),
    y3=np.round(bottomR[0], 0)
))
hover = HoverTool(
    tooltips=[
        ("Year: ", "@x{0 a}"),
        ("10% percentile: ", "@y1{0 a}"),
        ("Expected value: ", "@y2{0 a}"),
        ("90% percentile: ", "@y3{0 a}")
    ],
    mode='vline',
    names=['mean']
)
plot = figure(y_range=(0, 500), plot_width=500, plot_height=300, x_axis_label='Years',
              y_axis_label='Portfolio value, $m', tools=[hover])
plot.line('x', 'y1', source=sourcePlot, line_width=2, line_color="#2ca25f", legend="10% percentile", name='top')
meanGraph = plot.line('x', 'y2', source=sourcePlot, line_width=2, line_color="#2171b5",
                      legend="Expected value", name='mean')
plot.line('x', 'y3', source=sourcePlot, line_width=2, line_color="#e34a33", legend="90% percentile", name='bottom')
plot.legend.location = "top_left"
plot.legend.label_text_font_size = '8pt'

label1 = Label(x=265, y=90, x_units='screen', text=labelCAGR("10% perc CAGR"), render_mode='canvas',
               text_font_size='8pt')
label2 = Label(x=265, y=60, x_units='screen', text=labelCAGR("Mean CAGR"), render_mode='canvas', text_font_size='8pt')
label3 = Label(x=265, y=30, x_units='screen', text=labelCAGR("90% perc CAGR"), render_mode='canvas',
               text_font_size='8pt')
label4 = Label(x=368, y=90, x_units='screen', text=calcCAGR(topR[0][0], topR[0][-1]), render_mode='canvas',
               text_font_size='8pt')
label5 = Label(x=368, y=60, x_units='screen', text=calcCAGR(meanR[0][0], meanR[0][-1]), render_mode='canvas',
               text_font_size='8pt')
label6 = Label(x=368, y=30, x_units='screen', text=calcCAGR(bottomR[0][0], bottomR[0][-1]), render_mode='canvas',
               text_font_size='8pt')

plot.add_layout(label1)
plot.add_layout(label2)
plot.add_layout(label3)
plot.add_layout(label4)
plot.add_layout(label5)
plot.add_layout(label6)

sourceBar = ColumnDataSource(data=dict(
    x=durNames,
    y1=np.round(durR[0] * forecast.s0, 1),
    y2=np.round((durCompR[0] - durR[0]) * s0, 1)
))
bar = figure(x_range=durNames, y_range=(0, forecast.s0 * 1.2), title=None, plot_width=500, plot_height=200,
             y_axis_label='Available cash, $m', tools=[])
bar.vbar_stack(['y1', 'y2'], x='x', source=sourceBar, width=0.5, color=['#2171b5', '#deebf7'],
               legend=['Invested', 'Plus expected return'])
bar.legend.location = "top_left"
bar.legend.label_text_font_size = '8pt'
labelsBar = LabelSet(x='x', y='y1', text='y1', level='glyph', text_align='center', text_font_size='8pt',
                     x_offset=0, y_offset=5, source=sourceBar, render_mode='canvas')
bar.add_layout(labelsBar)


def getClbWeightChange(n):
    def closure(attr, oldW, newW):
        w[n][0] = newW / 100
        ds = np.round(sum(w) * 100, 0)[0] - 100
        if ds > 0:
            if n != 0:
                if w[0][0] >= ds / 100:
                    w[0][0] = w[0][0] - ds / 100
                else:
                    w[0][0] = 0
                    w[n][0] = w[n][0] - ds / 100 - w[0][0]
            else:
                w[0][0] = oldW / 100

        elif ds < 0:
            # if n != 0:
            w[0][0] = w[0][0] - ds / 100

        recalcGraphs()
        redraw()

    return closure


def recalcGraphs():
    topR, meanR, bottomR, durR, durCompR = forecast.calc(w)
    sourcePlot.data['y1'] = np.round(topR[0], 0)
    sourcePlot.data['y2'] = np.round(meanR[0], 0)
    sourcePlot.data['y3'] = np.round(bottomR[0], 0)
    sourceBar.data['y1'] = np.round(durR[0] * forecast.s0, 1)
    sourceBar.data['y2'] = np.round((durCompR[0] - durR[0]) * forecast.s0, 1)

    label4.text = calcCAGR(topR[0][0], topR[0][-1])
    label5.text = calcCAGR(meanR[0][0], meanR[0][-1])
    label6.text = calcCAGR(bottomR[0][0], bottomR[0][-1])


def clbS0Change(attr, old, newS0):
    forecast.s0 = newS0
    recalcGraphs()
    redraw()


def redraw():
    for ix, _ in enumerate(w[:]):
        sliders[ix].remove_on_change('value', callbacks[ix])
        sliders[ix].value = w[ix][0] * 100
        sliders[ix].on_change('value', callbacks[ix])
        divs[ix].text = """Amount: $""" + str(np.round(forecast.s0 * sliders[ix].value / 100, 1)) + """m"""


sliders = []
callbacks = []
for ix, assetName in enumerate(assets):
    sliders.append(Slider(start=0, end=100, value=int(w[0][0] * 100), step=1, title=assetName, bar_color='#2171b5'))
    callbacks.append(getClbWeightChange(ix))
    sliders[-1].on_change('value', callbacks[-1])
sliders[0].bar_color = 'lightgray'

initialAmountSlider = Slider(start=50, end=200, value=forecast.s0, step=1, title="Amount to be allocated, $m",
                             bar_color='#2171b5')
initialAmountSlider.on_change('value', clbS0Change)

p1 = Div(text="""<b>Projected portfolio growth</b>""")
p2 = Div(text="""<b>Split of portfolio per cash lockup duration</b>""")
# p3 = Div(text="""Total portfolio value: <b>$""" + str(forecast.s0) + """m</b>""")
p3 = Div(text="""Weights of asset classes:""")
pEmpty = Div(text=""" """)

divs = list(map(lambda s: Div(text="""Amount: $""" + str(np.round(forecast.s0 * s.value / 100, 1)) + """m"""), sliders))
rows = list(map(lambda r: list(r), zip(sliders, divs)))

layout = row(
    column(p1, plot, p2, bar),
    layout([initialAmountSlider], [pEmpty], [p3], *rows)
)

curdoc().add_root(layout)
