import pandas

algo = "bline"

for algo in ["bline", "ours"]:
    fname0 = f"/Users/tim/Data/nips_figures/XIRL/{algo}/gripperfromlongstick_0.csv"
    fname1 = f"/Users/tim/Data/nips_figures/XIRL/{algo}/gripperfromlongstick_1.csv"
    fname2 = f"/Users/tim/Data/nips_figures/XIRL/{algo}/gripperfromlongstick_2.csv"
    fnameout = f"/Users/tim/Data/nips_figures/XIRL/{algo}/gripperfromlongstick.csv"
    data0 = pandas.read_csv(fname0)
    data1 = pandas.read_csv(fname1)
    data2 = pandas.read_csv(fname2)

    data_all = pandas.concat((data0.iloc[:,1],
                              data0.iloc[:,2],data0.iloc[:,2],data0.iloc[:,2],data0.iloc[:,2],data0.iloc[:,2],data0.iloc[:,2],
                              data1.iloc[:,2],data1.iloc[:,2],data1.iloc[:,2],data1.iloc[:,2],data1.iloc[:,2],data1.iloc[:,2],
                              data2.iloc[:,2],data2.iloc[:,2],data2.iloc[:,2],data2.iloc[:,2],data2.iloc[:,2],data2.iloc[:,2]), axis=1)

    data_all.to_csv(f"/Users/tim/Data/nips_figures/XIRL/{algo}/gripperfromlongstick.csv", index=False)
