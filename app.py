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

# prediction function New!
def predict_new_rank(user_id_list, train_interaction, test_interaction, model):

    import datetime

    init_time = datetime.datetime.now()

    pred_interaction = np.zeros((train_interaction.shape[0], len(item_map)))

    n_user = len(user_map) # to be removed
    check_point = n_user//100  # to be removed
    interaction_cnt = test_interaction.sum(axis=1)

    for user_index, a_user in enumerate(user_id_list):

        if user_index%check_point == 0:
            print("{:.5%}".format(user_index/n_user)) # to be removed
            delta = datetime.datetime.now() - init_time
            print(f'Runtime {str(delta).split(".")[0]} Hour')

        if interaction_cnt[user_index] > 0:

            user_id = a_user  # to be decided whether user_id or user_name

            # Predict data as normal
            pred = model.predict(user_ids=user_id, item_ids=list(range(0, len(item_map), 1)))

            # Pull user input data
            user_input = train_interaction.todense()[user_id,] # All interactions of interested user
            user_input_txn = [i for i, x in enumerate(user_input.T) if x > 0] # All item index of a user txn

            # Query negative leverage items
            # Create a list of all combination

            import itertools
            all_txn_comb = list()
            for L in range(1, len(user_input_txn) + 1):
                for subset in itertools.combinations(user_input_txn, L):
                    all_txn_comb.append(set(subset))

            # query negative
            remove_set = set()
            for idx, row in df_negative.iterrows():
                for txn in all_txn_comb:
                    if txn == row['antecedents_set']:
                        remove_set = remove_set.union(item_mapping(row['consequents']))
            # query positive
            add_set = set()
            for idx, row in df_positive.iterrows():
                for txn in all_txn_comb:
                    if txn == row['antecedents_set']:
                        add_set = add_set.union(item_mapping(row['consequents']))
            new_pred = np.copy(pred) # to be removed

            # replace the prediction score with negative meaning value
            new_pred[list(add_set)] = 99
            new_pred[list(remove_set)] = 0 # to be desired on the value

            # create prediction interaction matrix
            indices = np.argsort(new_pred)[::-1]

            # predict rank
            tmp_interaction = indices.argsort()

            # Add result to final matrix
            pred_interaction[user_id] = tmp_interaction

        else:

            pred_interaction[a_user] = np.zeros((1, len(item_map)))

    from scipy import sparse

    pred_interaction = sparse.coo_matrix(pred_interaction)

    return pred_interaction

# Create function for features
def userInputFeatures():

    user_type = st.sidebar.radio("Please select user type.",('Existing User', 'New User', 'Cold-start'))

    form = st.sidebar.form('Form')

    if user_type == "Existing User":
        user_id = form.selectbox('Existing userid', [user for user, index in user_map.items()][:1000]) # Limit 1000 user

    elif user_type == "Cold-start":
        form.write('Please hit submit.')
        preference_dict = dict()

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
    elif user_type == "Cold-start" :
        new_user_preference = list()
    else:
        new_user_preference = list(data.iloc[0].values)


    return new_user_preference, user_type

def main():
    # Create sidebar
    st.sidebar.header('Input category preferences')

    new_user_preference, user_type = userInputFeatures()

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

    if user_type == "Cold-start":
        category_ranking = ['appliances.personal.massager', 'electronics.clocks',
         'electronics.audio.headphone', 'appliances.kitchen.refrigerators',
         'appliances.environment.vacuum', 'computers.peripherals.printer',
         'computers.desktop', 'electronics.camera.photo', 'electronics.video.tv',
         'electronics.camera.video', 'computers.components.cooler',
         'electronics.audio.subwoofer', 'electronics.smartphone',
         'computers.components.power_supply', 'electronics.tablet',
         'computers.notebook', 'electronics.audio.acoustic',
         'electronics.video.projector', 'electronics.audio.microphone',
         'electronics.audio.music_tools.piano', 'electronics.telephone',
         'computers.peripherals.mouse', 'computers.ebooks',
         'computers.components.motherboard', 'computers.components.videocards',
         'computers.peripherals.monitor', 'computers.peripherals.camera',
         'computers.components.memory', 'computers.components.cpu',
         'computers.peripherals.keyboard', 'computers.components.hdd',
         'computers.components.sound_card', 'computers.components.cdrw',
         'electronics.audio.dictaphone']

        final_prediction_list = [name for rank, name in enumerate(category_ranking) if rank < k]

    else:
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
                                , test_interaction=csr_matrix(temp_interaction)
                                , model=model)

        new_user_pred = pred.toarray()[-1]
        new_user_pred[new_user_preference_index] = 99

        # Recalculate Prediction Rank
        final_ranking = np.argsort(new_user_pred)

        predicted_item_id = [index for rank, index in enumerate(final_ranking) if rank < k]

        predicted_item_name = [name for name, index in item_map.items() if index in predicted_item_id]

        final_prediction_list = [predicted_item_name[i] for i in np.array(predicted_item_id).argsort()]

    st.write(final_prediction_list)

if __name__ == '__main__':
    main()
