# app.py
import streamlit as st
import pandas as pd
from st_clickable_images import clickable_images
from sklearn.neighbors import NearestNeighbors

# ------------------ Column Names ------------------ #
TITLE_COL = "title"
RATING_COL = "average_rating"
COUNT_COL = "rating_number"
PRICE_COL = "price"
IMAGE_COL = "image_url"
ASIN_COL = "parent_asin"

# ------------------ Load Data ------------------ #
@st.cache_data
def load_data():
    items = pd.read_csv("metadata.csv")
    interactions = pd.read_csv("reviews.csv")
    return items, interactions

items, interactions = load_data()

# ------------------ Popularity Score ------------------ #
pdata = items[[TITLE_COL, RATING_COL, COUNT_COL, PRICE_COL, IMAGE_COL, ASIN_COL]].copy()
n = round(pdata[COUNT_COL].mean(), 2)
pdata["PopularityScore"] = round((pdata[RATING_COL] * pdata[COUNT_COL]) / n, 2)

# ------------------ Content-based Data ------------------ #
cdata = items[[RATING_COL, COUNT_COL, PRICE_COL]].copy()
cosinemodel = NearestNeighbors(n_neighbors=6, metric="cosine")
cosinemodel.fit(cdata)

# ------------------ Functions ------------------ #
def recommend_similar_items(itemtitle, n=5):
    idx = items[items[TITLE_COL] == itemtitle].index
    if len(idx) == 0:
        return []
    idx = idx[0]
    data = cdata.iloc[idx]
    distances, indices = cosinemodel.kneighbors([data], n_neighbors=n + 1)
    return indices.flatten()[1:]


def show_product_grid(df, key_prefix="grid"):
    """Displays clickable product grid and returns selected row"""
    image_paths = list(df[IMAGE_COL])
    captions = [
        f"{t[:25]} | ⭐{r:.1f} | 💲{p}"
        for t, r, p in zip(df[TITLE_COL], df[RATING_COL], df[PRICE_COL])
    ]

    selected_index = clickable_images(
        image_paths,
        titles=captions,
        div_style={"display": "flex", "flex-wrap": "wrap", "gap": "25px"},
        img_style={
            "height": "150px",
            "width": "150px",
            "border-radius": "10px",
            "cursor": "pointer",
        },
        key=f"{key_prefix}_images",
    )

    if selected_index is not None and selected_index >= 0:
        return df.iloc[selected_index]
    return None


def render_stars(rating, max_stars=5):
    """Return star emoji string based on rating"""
    if pd.isna(rating):
        return ""
    full = int(rating)
    half = 1 if rating - full >= 0.5 else 0
    empty = max_stars - full - half
    return "⭐" * full + ("⭐" if half else "") + "☆" * empty


def show_selected_product(sel_row, interactions):
    """Show details of selected product with metadata + reviews"""
    st.divider()
    st.subheader("✅ Selected Product")

    # ---- Product Basic Info ----
    st.image(sel_row[IMAGE_COL], width=220)  # removed caption ✅

    st.markdown(
        f"""
        ### {sel_row[TITLE_COL]}
        ⭐ **{sel_row[RATING_COL]:.1f}** ({int(sel_row[COUNT_COL])} ratings)  
        💲 **Price:** {sel_row[PRICE_COL]}  
        🔑 **ASIN:** {sel_row[ASIN_COL]}
        """,
        unsafe_allow_html=True,
    )

    # ---- Extra Metadata ----
    if "main_category" in sel_row or "details" in sel_row:
        for col in ["main_category", "details"]:
            if col in sel_row and pd.notna(sel_row[col]):
                st.write(f"**{col.capitalize()}**: {sel_row[col]}")

    # ---- Show Reviews (Match by ASIN) ----
    st.markdown("### 📝 Customer Reviews")
    product_reviews = interactions[interactions[ASIN_COL] == sel_row[ASIN_COL]].head(5)

    if not product_reviews.empty:
        for _, row in product_reviews.iterrows():
            review_rating = (
                row["rating"] if "rating" in row and pd.notna(row["rating"]) else None
            )
            review_text = (
                row["text"] if "text" in row and pd.notna(row["text"]) else ""
            )
            review_title = (
                row["summary"] if "summary" in row and pd.notna(row["summary"]) else ""
            )

            stars_display = render_stars(review_rating) if review_rating else ""

            with st.container():
                st.markdown(
                    f"""
                    <div style="padding:12px; margin-bottom:12px; border-radius:10px;
                                background-color:#fdfdfd; border:1px solid #ccc;">
                        <span style="color:#e6b800; font-size:14px;">{stars_display}</span><br>
                        <b style="font-size:15px;">{review_title}</b><br>
                        <span style="color:#333; font-size:14px;">{review_text}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("No reviews available for this product.")


# ------------------ Helpers ------------------ #
def update_selected_product(sel_row):
    """Update selected product + ensure it's always moved to top of Popular section"""
    if sel_row is not None:
        # Always update selected product
        st.session_state.selected_product = sel_row

        # Remove product if already exists
        st.session_state.popular_dynamic = st.session_state.popular_dynamic[
            st.session_state.popular_dynamic[TITLE_COL] != sel_row[TITLE_COL]
        ]

        # Insert at top
        st.session_state.popular_dynamic = pd.concat(
            [sel_row.to_frame().T, st.session_state.popular_dynamic]
        ).head(12)


# ------------------ Streamlit UI ------------------ #
def main():
    st.set_page_config(page_title="All Beauty Recommender", layout="wide")
    st.subheader("💄 All Beauty Recommendation System")

    # Init session
    if "popular_dynamic" not in st.session_state:
        st.session_state.popular_dynamic = pdata.sort_values(
            by="PopularityScore", ascending=False
        ).head(12)

    if "selected_product" not in st.session_state:
        st.session_state.selected_product = None

    # 1. Popular Products
    st.subheader("🔥 Popular Beauty Products")
    sel_row = show_product_grid(st.session_state.popular_dynamic, key_prefix="popular")
    if sel_row is not None:  # ✅ only update when clicked
        update_selected_product(sel_row)

    # 2. Show Selected Product
    if st.session_state.selected_product is not None:
        sel_row = st.session_state.selected_product
        show_selected_product(sel_row, interactions)

        # 3. Content-based Recommendations
        st.subheader("🟢 Content-based Recommendations")
        recs = recommend_similar_items(st.session_state.selected_product[TITLE_COL], n=6)
        rec_df = items.iloc[recs][
            [TITLE_COL, RATING_COL, PRICE_COL, IMAGE_COL, COUNT_COL, ASIN_COL]
        ]
        rec_sel = show_product_grid(rec_df, key_prefix="content")
        if rec_sel is not None:  # ✅ only update when clicked
            update_selected_product(rec_sel)

        # 4. Collaborative Recommendations
        st.subheader("🔵 Collaborative Recommendations")
        collab = items.sort_values(by=RATING_COL, ascending=False).head(6)
        collab_sel = show_product_grid(collab, key_prefix="collab")
        if collab_sel is not None:  # ✅ only update when clicked
            update_selected_product(collab_sel)


# ------------------ Run ------------------ #
if __name__ == "__main__":
    main()

