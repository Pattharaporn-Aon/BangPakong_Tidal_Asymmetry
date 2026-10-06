# Notebook

`BangPakong_Tidal_Asymmetry.ipynb` walks through the whole analysis in numbered
steps, from reading the Excel file to the paired tests over the 26 withheld-data
experiments. It is the same code as `src/`, laid out for reading rather than for
batch running.

The notebook is also available on Colab at
https://colab.research.google.com/drive/1AjxrQJxLJqxxAJjhpjZsII2j68EnOKb5.

Upload `data/raw/Clean_BangPaKong_2.xlsx` when step 1 asks for it. Step 18 is
the slow one, at about three hours. Setting `CASES` near the top of that cell to a
short list such as `[0, 14, 19]` runs a few experiments first.
