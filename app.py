import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import vstack, csr_matrix
from lightfm import LightFM
import pickle


#load model, set cache to prevent reloading
@st.cache(allow_output_mutation=True)
def load_model():

    # Load development data
    interactions, df_positive, df_negative, catecory_df, item_map, user_map = pickle.load(open('variable_environment.pkl','rb'))
    userid_df = pd.read_csv('user_id.csv').dropna()

    # Init model
    model = LightFM(loss='logistic',
                    no_components=20,
                    learning_schedule='adagrad')

    # Train model
    model = model.fit(interactions,
                      epochs=64,
                      num_threads=7,
                      verbose=True)

    return model, interactions, df_positive, df_negative, catecory_df, item_map, user_map, userid_df

with st.spinner("Loading Model...."):
    model, interactions, df_positive, df_negative, catecory_df, item_map, user_map, userid_df = load_model()

def item_mapping(item_set):
    final_set = set()
    for item in item_set:
        final_set.add(item_map[item])
    return final_set

def predict_new_rank(user_id_list, train_interaction, test_interaction):

    pred_interaction = np.zeros((train_interaction.shape[0], train_interaction.shape[1]))
    interaction_cnt = test_interaction.sum(axis=1)

    for user_id in user_id_list:

        if interaction_cnt[user_id] > 0:

            # Predict data as normal
            pred = model.predict(user_ids=user_id, item_ids=list(range(0, len(item_map), 1)))

            # Pull user input data
            user_input = train_interaction.todense()[user_id,] # All interactions of interested user
            user_input_txn = [i for i, x in enumerate(user_input.T) if x > 0] # All item index of a user txn

            # Query negative leverage items
            # Create a list of all combination
            import itertools
            all_txn_comb = set()
            for L in range(1, len(user_input_txn) + 1):
                for subset in itertools.combinations(user_input_txn, L):
                    all_txn_comb = all_txn_comb.union({frozenset(subset)})

            # query negative
            remove_set = set()
            for idx, row in df_negative.iterrows():
                if row['antecedents_set'] in all_txn_comb:
                    remove_set = remove_set.union(item_mapping(row['consequents']))

            # query positive
            add_set = set()
            for idx, row in df_positive.iterrows():
                if row['antecedents_set'] in all_txn_comb:
                    add_set = add_set.union(item_mapping(row['consequents']))

            new_pred = np.copy(pred) # to be removed

            # replace the prediction score with negative meaning value
            new_pred[list(add_set)] = 99 # max score
            new_pred[list(remove_set)] = 0 # min score

            # create prediction interaction matrix
            indices = np.argsort(pred)[::-1]

            # predict rank
            tmp_interaction = indices.argsort()

            # Add result to final matrix
            pred_interaction[user_id] = tmp_interaction

        else:

            pred_interaction[user_id] = np.zeros((1, len(item_map)))

    from scipy import sparse

    pred_interaction = sparse.csr_matrix(pred_interaction)

    return pred_interaction

# Create function for features
def userInputFeatures():

    user_type = st.sidebar.radio("Please select user type.",('Existing User', 'New User'))

    form = st.sidebar.form('Form')

    if user_type == "Existing User":
        user_id = form.selectbox('Existing userid', [user for user, index in user_map.items()][:1000]) # Limit 1000 user

    else:
        # Create slider bar selection
        preference_cnt = form.slider('Number of Preference', 1, 10, step = 1)


        # Create dropdown of category
        preference_dict = dict()
        # category_list = category_df['category_code'].to_list()
        category_list = [name for name, index in item_map.items()]
        category_list.sort()

        for cnt in range(preference_cnt):
            tmp_category = form.selectbox(f'Category Preference{cnt+1}', category_list)
            preference_dict[f'category{cnt+1}'] = [tmp_category]

    submitted = form.form_submit_button('Submit')


    # Create dictonary data
    if user_type == "Existing User":
        data = {'user_id': [user_id]}
    else:
        data = preference_dict

    # Create dataframe
    data = pd.DataFrame(data)

    if user_type == "Existing User":
        user_preference_index = [index for index, value in enumerate(interactions.toarray()[user_map[data.loc[0,'user_id']],:]) if value == 1]
        new_user_preference = [category_code for category_code, index in item_map.items() if index in user_preference_index]
    else:
        new_user_preference = list(data.iloc[0].values)


    return new_user_preference

def main():
    # Create sidebar
    st.sidebar.header('Input category preferences')

    new_user_preference = userInputFeatures()

    # Header of page
    st.header(':orange[The ecommerce recommender system]')

    #Sub-header of page
    st.subheader('What is it?')

    # Body paragraph
    st.write(
        """
        This page is the app to predict your next purchase categories based on ecommerce transactional data from a multi-category ecommerce store.
        """
    )

    st.subheader('To use model prediction, please following below steps:')

    st.write(
        """
        1. Select existing userid or create your new userid. \n
        2. If you choose to create new userid please add you current preferences. \n
        3. See the recommended category below.
        """
    )

    st.subheader('Current user preferences')

    st.write(new_user_preference)

    st.subheader('Predicted user preferences')

    # Generate Prediction for k top categories
    k = st.slider("Please top k predictions.", 1, 10, step = 1)

    new_user_preference_index = [v for k, v in item_map.items() if k in new_user_preference]

    new_interaction = np.zeros([1, len(item_map)])
    new_interaction[:,new_user_preference_index] = 1

    new_coo = vstack([interactions, new_interaction])

    model.fit(interactions=new_coo)

    temp_interaction = np.zeros([len(user_map) + 1, len(item_map)])
    temp_interaction[-1] = 1

    # Predict using new logic
    pred = predict_new_rank(user_id_list=(range(0,len(user_map)+1,1))
                            , train_interaction=new_coo
                            , test_interaction=csr_matrix(temp_interaction))

    new_user_pred = pred.toarray()[-1]
    new_user_pred[new_user_preference_index] = 99

    # Recalculate Prediction Rank
    indices = np.argsort(new_user_pred)
    final_ranking = indices.argsort()

    predicted_item = [index for index, rank in enumerate(final_ranking) if rank < k]

    st.write([name for name, index in item_map.items() if index in predicted_item])

if __name__ == '__main__':
    main()