import pandas as pd
import matplotlib.pyplot as plt

Corr_data = pd.read_csv("C:/Users/AcquahEmmanuel/Desktop/Udemy DS/Section 3/Multiple Linear/ACF/03+-+corr.csv")
df = Corr_data['t0']
print(df)

# convert 't0' to float

df = pd.to_numeric(df, downcast='float')

# Plot ACF

plt.acorr(df, maxlags=10)
plt.show()


# Using Pandas shift function to creat a timelag data set
# t_1 and t_2

t_1 = df.shift(+1).to_frame()
t_2 = Corr_data['t0'].shift(+2).to_frame()