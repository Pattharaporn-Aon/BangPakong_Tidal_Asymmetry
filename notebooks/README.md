# Notebook

`BangPakong_Tidal_Asymmetry.ipynb` walks through the whole analysis in nineteen
steps, from reading the Excel file to the paired tests over the 26 withheld-data
experiments. It is the same code as `src/`, laid out for reading rather than for
batch running.

The notebook lives on Colab:
https://colab.research.google.com/drive/1AjxrQJxLJqxxAJjhpjZsII2j68EnOKb5

To put it in this folder, open it there and choose
**File → Save a copy in GitHub**, select this repository, and set the path to
`notebooks/BangPakong_Tidal_Asymmetry.ipynb`. Colab will ask for its own GitHub
authorisation; nothing needs to be pasted anywhere else.

Upload `data/raw/Clean_BangPaKong_2.xlsx` when step 1 asks for it. Step 18 is
the slow one, about three hours; set `CASES` near the top of that cell to a
short list such as `[0, 14, 19]` to try a few experiments first.
