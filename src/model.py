# Predictive Modelling Notes

# Simple Linear Regression
# Independent Variable vs Dependent Variable
# Independent Variables = 'open', 'high', 'low', 'volume' 
# Dependent Variables = 'close'

#Variables -- not sure if good to keep to calculate or just use df['column']
def x():
    return sum(df['low'])

def y():
    return sum(df['close'])

# Calculate mean
def mean_x():
    return sum(df['low'])/len(df['low'])

def mean_y():
    return y()/len(df['close'])
# Calculate x - mean-x, y - mean-y
def x_mean_x():
    pass

def y_mean_y():
    pass
# Calculate (x - mean-x)^2 and (x - mean-x)(y - mean-y)
# Calculate predicted y = b0-intercept + b1-slope * x
#   value of b1-slope: (sum(x - mean-x)(y - mean-y)) / sum(x - mean-x)^2
#   value of b0-intercept: using formula: predicted-y = b0-intercept + b1-slope * mean-x
# Draw best fit line
# Calculate R-squared
#   calculate SSE & SSR

# Check Correlation between variables
variables =  ['close', 'open', 'low', 'high', 'volume']
corr_matrix = df[variables].corr()
print
# Since 'low' has the highest correlation, we will take 'low' as the independent variable

print('Mean of x is: ',mean_x())
print('Mean of y is: ',mean_y())