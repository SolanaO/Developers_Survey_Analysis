{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# create a function that will count the type of strings in a column\n",
    "def counts_strings(strings_list, dframe, incol):\n",
    "    \"\"\"\n",
    "    Counts the number of occurences of a given string among the \n",
    "    text entries of a column of a dataframe. \n",
    "    INPUT: \n",
    "        strings_list = list of strings to search for\n",
    "        dframe = pandas dataframe\n",
    "        incol = the column where to search for strings\n",
    "    OUTPUT:\n",
    "        pandas dataframe with two columns\n",
    "    \"\"\"\n",
    "    my_counts = defaultdict(int)\n",
    "    for entry in strings_list:\n",
    "        my_counts[entry] = dframe[incol].str.contains(entry).sum()\n",
    "    new_df = pd.DataFrame.from_dict(my_counts, orient = 'index').reset_index()\n",
    "    return new_df"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
