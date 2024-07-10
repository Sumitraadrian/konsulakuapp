# Muat model lama
from tensorflow.keras.models import load_model
old_model = load_model('model/skin_disease_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})

# Simpan ulang dalam format SavedModel
old_model.save('model/skin_disease_model_saved')
