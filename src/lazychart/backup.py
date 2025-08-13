
## trash

def line():

    if trendline == 'linear':
        try:
            if pd.api.types.is_datetime64_any_dtype(s[x]):
                xnum = s[x].astype('int64') // 10**9
            else:
                xnum = pd.to_numeric(s[x])
            coeffs = np.polyfit(xnum, s[y].astype(float), 1)
            poly = np.poly1d(coeffs)
            if forecast_steps:
                step = (xnum.max() - xnum.min())/(len(xnum)-1) if len(xnum)>1 else 1
                ext_x = np.concatenate([xnum, xnum.max() + np.arange(1, forecast_steps+1)*step])
                ys = poly(ext_x)
                if pd.api.types.is_datetime64_any_dtype(s[x]):
                    ext_x_dt = pd.to_datetime(ext_x*10**9)
                    ax.plot(ext_x_dt, ys, linestyle='--', linewidth=1)
                else:
                    ax.plot(ext_x, ys, linestyle='--', linewidth=1)
            else:
                xs_sorted = np.sort(xnum)
                ax.plot(pd.to_datetime(xs_sorted*10**9) if pd.api.types.is_datetime64_any_dtype(s[x]) else xs_sorted, np.polyval(coeffs, xs_sorted), linestyle='--', linewidth=1)

def scatter():
    
    df = data.copy()
    fig, ax = plt.subplots(figsize=figsize)
    xs = pd.to_numeric(df[x], errors='coerce')
    ys = pd.to_numeric(df[y], errors='coerce')

    if jitter and not pd.api.types.is_numeric_dtype(df[x]):
        cats = df[x].astype('category')
        xnum = cats.cat.codes.astype(float)
        xnum = xnum + np.random.normal(0, jitter, size=len(df))
        xs = xnum
        ax.set_xticks(range(len(cats.cat.categories)))
        ax.set_xticklabels(cats.cat.categories, rotation=45, ha='right')

    sizes = None
    if size and size in df.columns:
        sizes = (pd.to_numeric(df[size], errors='coerce').fillna(1).values - pd.to_numeric(df[size], errors='coerce').min() + 1) * 10

    colors = None
    if hue and hue in df.columns:
        cats = df[hue].astype('category')
        pal = cycle_palette(self.default_palette, len(cats.cat.categories))
        color_map = {cat: pal[i % len(pal)] for i, cat in enumerate(cats.cat.categories)}
        colors = df[hue].map(color_map).values

    ax.scatter(xs, ys, s=sizes, c=colors if colors is not None else color)

    if highlight and highlight in df.columns:
        mask = df[highlight].astype(bool)
        if mask.any():
            ax.scatter(xs[mask], ys[mask], facecolors='none', edgecolors=highlight_color, linewidths=1.5, s=(sizes[mask] if sizes is not None else 40))

    if annotate and annotate in df.columns:
        for xi, yi, txt in zip(xs, ys, df[annotate]):
            ax.annotate(str(txt), (xi, yi), fontsize=7, alpha=0.8)

def hist():
    pass
    
def pie():
    pass