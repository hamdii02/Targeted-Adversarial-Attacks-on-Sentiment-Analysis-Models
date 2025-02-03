from src.chameleon.models import HuggingFaceModel
from src.chameleon.probes import TextAttackProbe

# Load a HuggingFace model
model = HuggingFaceModel("lxyuan/distilbert-base-multilingual-cased-sentiments-student")
print(":::::::::::::::::::::")
# Define a probe
probe = TextAttackProbe(model, "My grandmother's secret sauce is the best ever made!")

# Run the probe
result = probe.run()

# Print results
print("Found sentence:", result.sentence)
print("Scores:", result.scores)
print("Elapsed time:", result.elapsed_time)