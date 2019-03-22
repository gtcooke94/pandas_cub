import numpy as np
from collections import defaultdict

__version__ = '0.0.1'


class DataFrame:

    def __init__(self, data):
        """
        A DataFrame holds two dimensional heterogeneous data. Create it by
        passing a dictionary of NumPy arrays to the values parameter

        Parameters
        ----------
        data: dict
            A dictionary of strings mapped to NumPy arrays. The key will
            become the column name.
        """
        # check for correct input types
        self._check_input_types(data)

        # check for equal array lengths
        self._check_array_lengths(data)

        # convert unicode arrays to object
        self._data = self._convert_unicode_to_object(data)

        # Allow for special methods for strings
        self.str = StringMethods(self)
        self._add_docs()

    def _check_input_types(self, data):
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")
        for key, value in data.items():
            if not isinstance(key, str):
                raise TypeError("Keys must be strings")
            if not isinstance(value, np.ndarray):
                raise TypeError("Values must be numpy arrays")
            if len(value.shape) != 1:
                raise ValueError(f"array for key {key} is not one-dimensional")

    def _check_array_lengths(self, data):
        it = iter(data.items())
        a_length = len(next(it)[1])
        all_same_length = all(a_length == len(i[1]) for i in it)
        if not all_same_length:
            raise ValueError("Input data not of equal length")

    def _convert_unicode_to_object(self, data):
        new_data = {k: (d.astype(object) if d.dtype.kind == 'U' else d) for
                    k, d in data.items()}
        return new_data

    def __len__(self):
        """
        Make the builtin len function work with our dataframe

        Returns
        -------
        int: the number of rows in the dataframe
        """
        return len(next(iter(self._data.items()))[1])

    @property
    def columns(self):
        """
        _data holds column names mapped to arrays
        take advantage of internal ordering of dictionaries to
        put columns in correct order in list. Only works in 3.6+

        Returns
        -------
        list of column names
        """
        return [k for k, _ in self._data.items()]

    @columns.setter
    def columns(self, columns):
        """
        Must supply a list of columns as strings the same length
        as the current DataFrame

        Parameters
        ----------
        columns: list of strings

        Returns
        -------
        None
        """
        if not isinstance(columns, list):
            raise TypeError("columns must be a list")
        if len(columns) != len(self._data):
            raise ValueError("New columns list must be of the same size as the"
                             " current number of columns")
        if any(not isinstance(c, str) for c in columns):
            raise TypeError("Columns contain a non-string object")
        if len(columns) != len(set(columns)):
            raise ValueError("There are duplicated values in the columns")
        self._data = {c: d[1] for c, d in zip(columns, self._data.items())}

    @property
    def shape(self):
        """
        Returns
        -------
        two-item tuple of number of rows and columns
        """
        return (len(self), len(self._data))

    def _repr_html_(self):
        """
        Used to create a string of HTML to nicely display the DataFrame
        in a Jupyter Notebook. Different string formatting is used for
        different data types.

        The structure of the HTML is as follows:
        <table>
            <thead>
                <tr>
                    <th>data</th>
                    ...
                    <th>data</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>{i}</strong></td>
                    <td>data</td>
                    ...
                    <td>data</td>
                </tr>
                ...
                <tr>
                    <td><strong>{i}</strong></td>
                    <td>data</td>
                    ...
                    <td>data</td>
                </tr>
            </tbody>
        </table>
        """
        html = '<table><thead><tr><th></th>'
        for col in self.columns:
            html += f"<th>{col:10}</th>"

        html += '</tr></thead>'
        html += "<tbody>"

        only_head = False
        num_head = 10
        num_tail = 10
        if len(self) <= 20:
            only_head = True
            num_head = len(self)

        for i in range(num_head):
            html += f'<tr><td><strong>{i}</strong></td>'
            for col, values in self._data.items():
                kind = values.dtype.kind
                if kind == 'f':
                    html += f'<td>{values[i]:10.3f}</td>'
                elif kind == 'b':
                    html += f'<td>{values[i]}</td>'
                elif kind == 'O':
                    v = values[i]
                    if v is None:
                        v = 'None'
                    html += f'<td>{v:10}</td>'
                else:
                    html += f'<td>{values[i]:10}</td>'
            html += '</tr>'

        if not only_head:
            html += '<tr><strong><td>...</td></strong>'
            for i in range(len(self.columns)):
                html += '<td>...</td>'
            html += '</tr>'
            for i in range(-num_tail, 0):
                html += f'<tr><td><strong>{len(self) + i}</strong></td>'
                for col, values in self._data.items():
                    kind = values.dtype.kind
                    if kind == 'f':
                        html += f'<td>{values[i]:10.3f}</td>'
                    elif kind == 'b':
                        html += f'<td>{values[i]}</td>'
                    elif kind == 'O':
                        v = values[i]
                        if v is None:
                            v = 'None'
                        html += f'<td>{v:10}</td>'
                    else:
                        html += f'<td>{values[i]:10}</td>'
                html += '</tr>'

        html += '</tbody></table>'
        return html
        pass

    @property
    def values(self):
        """
        Returns
        -------
        A single 2D NumPy array of the underlying data
        """
        return np.column_stack([d for _, d in self._data.items()])

    @property
    def dtypes(self):
        """
        Returns
        -------
        A two-column DataFrame of column names in one column and
        their data type in the other
        """
        DTYPE_NAME = {'O': 'string', 'i': 'int', 'f': 'float', 'b': 'bool'}
        col_names = np.array(self.columns)
        dtypes = np.array([DTYPE_NAME.get(self._data[k].dtype.kind) for k in
                           self.columns])
        return DataFrame({'Column Name': col_names, 'Data Type': dtypes})

    def __getitem__(self, item):
        """
        Use the brackets operator to simultaneously select rows and columns
        A single string selects one column -> df['colname']
        A list of strings selects multiple columns -> df[['colname1', 'colname2']]
        A one column DataFrame of booleans that filters rows -> df[df_bool]
        Row and column selection simultaneously -> df[rs, cs]
            where cs and rs can be integers, slices, or a list of integers
            rs can also be a one-column boolean DataFrame

        Returns
        -------
        A subset of the original DataFrame
        """
        if isinstance(item, str):
            return DataFrame({item: self._data[item]})
        elif isinstance(item, list):
            return DataFrame({i: self._data[i] for i in item})
        elif isinstance(item, DataFrame):
            if len(item.columns) != 1:
                raise ValueError("Boolean Dataframe input must be one column")
            _, inds = next(iter(item._data.items()))
            if inds.dtype.kind != 'b':
                raise ValueError('Index must be boolean')
            # Get just the rows that are true in the boolean array
            # Create the dict we will have with empty np array to append to
            index_rows = self.values[inds]
            indexed_dict = {c: index_rows[:, i] for i, c in
                            enumerate(self.columns)}
            return DataFrame(indexed_dict)
        elif not isinstance(item, tuple):
            raise TypeError(
                "Must either pass string, list of string, one column "
                "boolean DataFrame, or both a row and a column selection"
            )
        return self._getitem_tuple(item)

    def _getitem_tuple(self, item):
        # simultaneous selection of rows and cols -> df[rs, cs]
        if len(item) != 2:
            raise ValueError("simultaneous row and column selection must be of"
                             " length 2")
        row_selection, col_selection = item
        if isinstance(col_selection, int):
            col_selection = [self.columns[col_selection]]
        elif isinstance(col_selection, str):
            col_selection = [col_selection]
        elif isinstance(col_selection, list):
            col_selection = [c if isinstance(c, str) else self.columns[c]
                             for c in col_selection]
        elif isinstance(col_selection, slice):
            start = col_selection.start
            stop = col_selection.stop
            step = col_selection.step
            if isinstance(start, str):
                start = self.columns[start]
            if isinstance(stop, str):
                stop = self.columns[stop + 1]
            col_selection = self.columns[start:stop:step]

        else:
            raise TypeError("Column selection must be an integer, string,"
                            " list, or slice")

        if isinstance(row_selection, int):
            row_selection = [row_selection]
        elif isinstance(row_selection, DataFrame):
            if row_selection.shape[1] != 1:
                raise ValueError("row_selection dataframe must be one column")
            row_selection = row_selection.values
            if row_selection.dtype.kind != 'b':
                raise TypeError("Row selection DataFrame must be boolean")
        elif (not isinstance(row_selection, list) and not
              isinstance(row_selection, slice)):
            raise TypeError("row_selection must be an integer, list, slice, or"
                            " DataFrame")
        indexed_dict = {c: self[c].values[row_selection].flatten() for c in
                        col_selection}
        return DataFrame(indexed_dict)

    def _ipython_key_completions_(self):
        # allows for tab completion when doing df['c
        return self.columns

    def __setitem__(self, key, value):
        # adds a new column or a overwrites an old column
        if not isinstance(key, str):
            raise NotImplementedError("DataFrame can only set a single column")
        if isinstance(value, np.ndarray):
            if value.ndim != 1:
                raise ValueError("Value for new column must be a 1D numpy "
                                 "array")
            elif len(value) != len(self):
                raise ValueError("New column not same length as dataframe")
        elif isinstance(value, DataFrame):
            if len(value.columns) != 1:
                raise ValueError("Input must be a single column")
            elif len(value) != len(self):
                raise ValueError("New column not same length as dataframe")
            value = value.values
        elif isinstance(value, (str, int, float, bool)):
            value = np.repeat(value, len(self))
        else:
            raise TypeError("Input value not an accepted type")
        if value.dtype.kind == 'U':
            value = value.astype('O')
        self._data[key] = value

    def head(self, n=5):
        """
        Return the first n rows

        Parameters
        ----------
        n: int

        Returns
        -------
        DataFrame
        """
        return self[:n, :]

    def tail(self, n=5):
        """
        Return the last n rows

        Parameters
        ----------
        n: int

        Returns
        -------
        DataFrame
        """
        return self[-n:, :]

    #### Aggregation Methods ####

    def min(self):
        return self._agg(np.min)

    def max(self):
        return self._agg(np.max)

    def mean(self):
        return self._agg(np.mean)

    def median(self):
        return self._agg(np.median)

    def sum(self):
        return self._agg(np.sum)

    def var(self):
        return self._agg(np.var)

    def std(self):
        return self._agg(np.std)

    def all(self):
        return self._agg(np.all)

    def any(self):
        return self._agg(np.any)

    def argmax(self):
        return self._agg(np.argmax)

    def argmin(self):
        return self._agg(np.argmin)

    def _agg(self, aggfunc):
        """
        Generic aggregation function that applies the
        aggregation to each column

        Parameters
        ----------
        aggfunc: str of the aggregation function name in NumPy

        Returns
        -------
        A DataFrame
        """
        new_data = {}
        for col in self.columns:
            if self._data[col].dtype.kind == 'O':
                try:
                    data = aggfunc(self._data[col])
                except TypeError:
                    continue
            else:
                data = aggfunc(self._data[col])
            new_data[col] = (np.array(data)).flatten()
        return DataFrame(new_data)

    def isna(self):
        """
        Determines whether each value in the DataFrame is missing or not

        Returns
        -------
        A DataFrame of booleans the same size as the calling DataFrame
        """
        new_data = {c: np.isnan(self._data[c]) if self._data[c].dtype.kind !=
                    'O' else self._data[c] is None for c in self.columns}
        return DataFrame(new_data)

    def count(self):
        """
        Counts the number of non-missing values per column

        Returns
        -------
        A DataFrame
        """
        isna_frame = self.isna()
        sum_data = {c: np.array([len(self) - sum(isna_frame._data[c])]) for c
                    in self.columns}
        return DataFrame(sum_data)

    def unique(self):
        """
        Finds the unique values of each column

        Returns
        -------
        A list of one-column DataFrames
        """
        to_return = [DataFrame({c: np.unique(self._data[c])}) for c in
                     self.columns]
        return to_return[0] if len(to_return) == 1 else to_return

    def nunique(self):
        """
        Find the number of unique values in each column

        Returns
        -------
        A DataFrame
        """
        return DataFrame({c: np.array([len(np.unique(self._data[c]))]) for c in
                          self.columns})

    def value_counts(self, normalize=False):
        """
        Returns the frequency of each unique value for each column

        Parameters
        ----------
        normalize: bool
            If True, returns the relative frequencies (percent)

        Returns
        -------
        A list of DataFrames or a single DataFrame if one column
        """
        to_return = []
        col_values_count = {c: np.unique(self._data[c], return_counts=True) for
                            c in self.columns}
        for col, (values, counts) in col_values_count.items():
            order = np.argsort(counts)[::-1]
            if normalize:
                counts = counts / sum(counts)
            df_to_add = DataFrame({col: values[order], 'count': counts[order]})
            to_return.append(df_to_add)
        return to_return[0] if len(to_return) == 1 else to_return

    def rename(self, columns):
        """
        Renames columns in the DataFrame

        Parameters
        ----------
        columns: dict
            A dictionary mapping the old column name to the new column name

        Returns
        -------
        A DataFrame
        """
        if not isinstance(columns, dict):
            return ValueError("columns must be a dictionary")
        new_dict = {}
        for old_col in self.columns:
            new_col = columns.get(old_col, old_col)
            new_dict[new_col] = self._data[old_col]
        return DataFrame(new_dict)

    def drop(self, columns):
        """
        Drops one or more columns from a DataFrame

        Parameters
        ----------
        columns: str or list of strings

        Returns
        -------
        A DataFrame
        """
        if isinstance(columns, str):
            columns = [columns]
        elif not isinstance(columns, list):
            raise TypeError('columns must be a string or list of strings')

        new_data = {c: self._data[c] for c in self.columns if c not in columns}
        return DataFrame(new_data)

    #### Non-Aggregation Methods ####

    def abs(self):
        """
        Takes the absolute value of each value in the DataFrame

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.abs)

    def cummin(self):
        """
        Finds cumulative minimum by column

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.minimum.accumulate)

    def cummax(self):
        """
        Finds cumulative maximum by column

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.maximum.accumulate)

    def cumsum(self):
        """
        Finds cumulative sum by column

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.cumsum)

    def clip(self, lower=None, upper=None):
        """
        All values less than lower will be set to lower
        All values greater than upper will be set to upper

        Parameters
        ----------
        lower: number or None
        upper: number or None

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.clip, a_min=lower, a_max=upper)

    def round(self, n):
        """
        Rounds values to the nearest n decimals

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.round, decimals=n)

    def copy(self):
        """
        Copies the DataFrame

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.copy)

    def _non_agg(self, funcname, **kwargs):
        """
        Generic non-aggregation function
    
        Parameters
        ----------
        funcname: str of NumPy name
        kwargs: extra keyword arguments for certain functions

        Returns
        -------
        A DataFrame
        """
        return DataFrame({c: funcname(self._data[c], **kwargs) for c in
                         self.columns})

    def diff(self, n=1):
        """
        Take the difference between the current value and
        the nth value above it.

        Parameters
        ----------
        n: int

        Returns
        -------
        A DataFrame
        """
        def func(arr):
            if arr.dtype.kind == 'i':
                arr = arr.astype('f')
            right = np.roll(arr, n)
            left = arr
            result = left - right
            if n > 0:
                result[:n] = np.nan
            elif n < 0:
                result[n:] = np.nan
            return result
            
        return self._non_agg(func)

    def pct_change(self, n=1):
        """
        Take the percentage difference between the current value and
        the nth value above it.

        Parameters
        ----------
        n: int

        Returns
        -------
        A DataFrame
        """
        def func(arr):
            if arr.dtype.kind == 'i':
                arr = arr.astype('f')
            right = np.roll(arr, n)
            left = arr
            result = (left - right) / right
            if n > 0:
                result[:n] = np.nan
            elif n < 0:
                result[n:] = np.nan
            return result
        return self._non_agg(func)

    #### Arithmetic and Comparison Operators ####

    def __add__(self, other):
        return self._oper('__add__', other)

    def __radd__(self, other):
        return self._oper('__radd__', other)

    def __sub__(self, other):
        return self._oper('__sub__', other)

    def __rsub__(self, other):
        return self._oper('__rsub__', other)

    def __mul__(self, other):
        return self._oper('__mul__', other)

    def __rmul__(self, other):
        return self._oper('__rmul__', other)

    def __truediv__(self, other):
        return self._oper('__truediv__', other)

    def __rtruediv__(self, other):
        return self._oper('__rtruediv__', other)

    def __floordiv__(self, other):
        return self._oper('__floordiv__', other)

    def __rfloordiv__(self, other):
        return self._oper('__rfloordiv__', other)

    def __pow__(self, other):
        return self._oper('__pow__', other)

    def __rpow__(self, other):
        return self._oper('__rpow__', other)

    def __gt__(self, other):
        return self._oper('__gt__', other)

    def __lt__(self, other):
        return self._oper('__lt__', other)

    def __ge__(self, other):
        return self._oper('__ge__', other)

    def __le__(self, other):
        return self._oper('__le__', other)

    def __ne__(self, other):
        return self._oper('__ne__', other)

    def __eq__(self, other):
        return self._oper('__eq__', other)

    def _oper(self, op, other):
        """
        Generic operator function

        Parameters
        ----------
        op: str name of special method
        other: the other object being operated on

        Returns
        -------
        A DataFrame
        """
        if isinstance(other, DataFrame):
            if len(other.columns) != 1:
                raise ValueError(
                    f"{op.__name__} must either use a single value or a "
                    "single column DataFrame."
                )
            other = next(iter(self._data.values()))
        new_data = {c: getattr(self._data[c], op)(other) for c in
                    self.columns}
        return DataFrame(new_data)

    def sort_values(self, by, asc=True):
        """
        Sort the DataFrame by one or more values

        Parameters
        ----------
        by: str or list of column names
        asc: boolean of sorting order

        Returns
        -------
        A DataFrame
        """
        if isinstance(by, str):
            order = np.argsort(self._data[by])
        elif isinstance(by, list):
            order = np.lexsort([self._data[c] for c in by[::-1]])
        else:
            return ValueError("by must be a string or list of string")

        if not asc:
            order = order[::-1]
        ordered_data = {c: self._data[c][order] for c in self.columns}
        return DataFrame(ordered_data)


    def sample(self, n=None, frac=None, replace=False, seed=None):
        """
        Randomly samples rows the DataFrame

        Parameters
        ----------
        n: int
            number of rows to return
        frac: float
            Proportion of the data to sample
        replace: bool
            Whether or not to sample with replacement
        seed: int
            Seeds the random number generator

        Returns
        -------
        A DataFrame
        """
        if frac:
            if frac < 0:
                raise ValueError("Fraction must be positive")
            n = int(len(self) * frac)
        if not isinstance(n, int):
            raise TypeError("n must be an integer")
        if seed is not None:
            np.random.seed(seed)
        selection = np.random.choice(len(self), n, replace=replace)
        return self[list(selection), :]

    def pivot_table(self, rows=None, columns=None, values=None, aggfunc=None):
        """
        Creates a pivot table from one or two 'grouping' columns.

        Parameters
        ----------
        rows: str of column name to group by
            Optional
        columns: str of column name to group by
            Optional
        values: str of column name to aggregate
            Required
        aggfunc: str of aggregation function

        Returns
        -------
        A DataFrame
        """
        pivot_dict = {}
        if not columns:
            if not values:
                value_counts = self[rows].value_counts()
                return value_counts.rename({'count': 'size'})
            else:
                row_val_dict = self._get_row_val_dict(rows, values)
                to_return_dict = {
                    k: getattr(v, aggfunc)() for k, v in
                    row_val_dict.items()
                }
                rows_vals = [k for k in to_return_dict.keys()]
                values_vals = [v for v in to_return_dict.values()]
                pivot_dict = {rows: np.array(rows_vals), aggfunc:
                              np.array(values_vals)}
        elif not rows:
            if not values:
                value_counts = self[columns].value_counts()
                rows_vals = value_counts[columns].values.flatten()
                values_vals = value_counts['count'].values.flatten()
                pivot_dict = {row_name: count for row_name, count in
                              zip(rows_vals, values_vals)}

                pivot_dict = {row_name: np.array([count]) for row_name, count
                              in sorted(zip(rows_vals, values_vals))}
            else:
                row_val_dict = self._get_row_val_dict(columns, values)
                pivot_dict = {k: np.array([getattr(v, aggfunc)()]) for k, v in
                              row_val_dict.items()}
        else:
            # Full pivot implementation
            row_val_col_dict = self._get_row_val_col_dict(rows, columns,
                                                          values)
            pivot_columns = list(self[columns].unique().values.flatten())
            final_colums = [rows, *pivot_columns]
            final_row_values = self[rows].unique().values.flatten()
            pivot_dict = {rows: final_row_values}
            for c in pivot_columns:
                column_vals = []
                for row_val in final_row_values:
                    vals = np.array(row_val_col_dict[(row_val, c)])
                    column_vals.append(getattr(vals, aggfunc)())
                pivot_dict[c] = np.array(column_vals)

        return DataFrame(pivot_dict)

    def _get_row_val_col_dict(self, rows, columns, values):
        row_val_col_dict = defaultdict(list)
        for i in range(len(self)):
            row = self[i, [rows, columns, values]]
            row_val = row[rows].values.flatten()[0]
            col_val = row[columns].values.flatten()[0]
            value_val = row[values].values.flatten()[0]
            row_val_col_dict[(row_val, col_val)].append(value_val)
        return row_val_col_dict

    def _get_row_val_dict(self, rows, values):
        row_val_dict = {val[0]: np.array([]) for val in
                        self[rows].unique().values}
        for i in range(len(self)):
            row = self[i, [rows, values]].values
            row_val = row[0][0]
            value = row[0][1]
            row_val_dict[row_val] = np.append(row_val_dict[row_val], value)
        return row_val_dict


    def _add_docs(self):
        agg_names = ['min', 'max', 'mean', 'median', 'sum', 'var',
                     'std', 'any', 'all', 'argmax', 'argmin']
        agg_doc = \
        """
        Find the {} of each column
        
        Returns
        -------
        DataFrame
        """
        for name in agg_names:
            getattr(DataFrame, name).__doc__ = agg_doc.format(name)


class StringMethods:

    def __init__(self, df):
        self._df = df

    def capitalize(self, col):
        return self._str_method(str.capitalize, col)

    def center(self, col, width, fillchar=None):
        if fillchar is None:
            fillchar = ' '
        return self._str_method(str.center, col, width, fillchar)

    def count(self, col, sub, start=None, stop=None):
        return self._str_method(str.count, col, sub, start, stop)

    def endswith(self, col, suffix, start=None, stop=None):
        return self._str_method(str.endswith, col, suffix, start, stop)

    def startswith(self, col, suffix, start=None, stop=None):
        return self._str_method(str.startswith, col, suffix, start, stop)

    def find(self, col, sub, start=None, stop=None):
        return self._str_method(str.find, col, sub, start, stop)

    def len(self, col):
        return self._str_method(str.__len__, col)

    def get(self, col, item):
        return self._str_method(str.__getitem__, col, item)

    def index(self, col, sub, start=None, stop=None):
        return self._str_method(str.index, col, sub, start, stop)

    def isalnum(self, col):
        return self._str_method(str.isalnum, col)

    def isalpha(self, col):
        return self._str_method(str.isalpha, col)

    def isdecimal(self, col):
        return self._str_method(str.isdecimal, col)

    def islower(self, col):
        return self._str_method(str.islower, col)

    def isnumeric(self, col):
        return self._str_method(str.isnumeric, col)

    def isspace(self, col):
        return self._str_method(str.isspace, col)

    def istitle(self, col):
        return self._str_method(str.istitle, col)

    def isupper(self, col):
        return self._str_method(str.isupper, col)

    def lstrip(self, col, chars):
        return self._str_method(str.lstrip, col, chars)

    def rstrip(self, col, chars):
        return self._str_method(str.rstrip, col, chars)

    def strip(self, col, chars):
        return self._str_method(str.strip, col, chars)

    def replace(self, col, old, new, count=None):
        if count is None:
            count = -1
        return self._str_method(str.replace, col, old, new, count)

    def swapcase(self, col):
        return self._str_method(str.swapcase, col)

    def title(self, col):
        return self._str_method(str.title, col)

    def lower(self, col):
        return self._str_method(str.lower, col)

    def upper(self, col):
        return self._str_method(str.upper, col)

    def zfill(self, col, width):
        return self._str_method(str.zfill, col, width)

    def encode(self, col, encoding='utf-8', errors='strict'):
        return self._str_method(str.encode, col, encoding, errors)

    def _str_method(self, method, col, *args):
        pass


def read_csv(fn):
    """
    Read in a comma-separated value file as a DataFrame

    Parameters
    ----------
    fn: string of file location

    Returns
    -------
    A DataFrame
    """
    pass
