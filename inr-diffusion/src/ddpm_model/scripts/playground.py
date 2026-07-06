import cbct_artifact_reduction.dataprocessing as dp
import cbct_artifact_reduction.utils as utils

axeosIds = utils.getAllAxeosIDs()
accuitomoIds = utils.getAllAccuitomoIDs()
planmecaIds = utils.getAllplanmecaIDs()
x800Ids = utils.getAllx800IDs()

sample_axoes = axeosIds[0]
sample_accuitomo = accuitomoIds[0]
sample_planmeca = planmecaIds[0]
sample_x800 = x800Ids[0]

basepath = utils.ROOT_DIR
# print(basepath)
# sample_axoes_np = dp.single_nifti_to_numpy(
#     f"{basepath}/output/resized/256x256/{sample_axoes}.nii.gz"
# )
# sample_accuitomo_np = dp.single_nifti_to_numpy(
#     f"{basepath}/output/resized/256x256/{sample_accuitomo}.nii.gz"
# )
# sample_planmeca_np = dp.single_nifti_to_numpy(
#     f"{basepath}/output/resized/256x256/{sample_planmeca}.nii.gz"
# )
# sample_x800_np = dp.single_nifti_to_numpy(
#     f"{basepath}/output/resized/256x256/{sample_x800}.nii.gz"
# )
#
# print(f"Shape of Axeos image: {sample_axoes_np.shape}")
# print(f"Shape of Accuitomo image: {sample_accuitomo_np.shape}")
# print(f"Shape of Planmeca image: {sample_planmeca_np.shape}")
# print(f"Shape of x800 image: {sample_x800_np.shape}")

shapesAxeos = []
for i in axeosIds:
    a = dp.single_nifti_to_numpy(f"{basepath}/output/resized/256x256/{i}.nii.gz")
    shapesAxeos.append(a.shape[2])


print(shapesAxeos)

# do the same for the other scanners
shapesAccuitomo = []
for i in accuitomoIds:
    a = dp.single_nifti_to_numpy(f"{basepath}/output/resized/256x256/{i}.nii.gz")
    shapesAccuitomo.append(a.shape[2])

shapesPlanmeca = []
for i in planmecaIds:
    a = dp.single_nifti_to_numpy(f"{basepath}/output/resized/256x256/{i}.nii.gz")
    shapesPlanmeca.append(a.shape[2])

shapesX800 = []
for i in x800Ids:
    a = dp.single_nifti_to_numpy(f"{basepath}/output/resized/256x256/{i}.nii.gz")
    shapesX800.append(a.shape[2])

print(shapesAccuitomo)
print(shapesPlanmeca)
print(shapesX800)

# calculate the average number of frames for each scanner
avgAxeos = sum(shapesAxeos) / len(shapesAxeos)
avgAccuitomo = sum(shapesAccuitomo) / len(shapesAccuitomo)
avgPlanmeca = sum(shapesPlanmeca) / len(shapesPlanmeca)
avgX800 = sum(shapesX800) / len(shapesX800)

# Print all averages
