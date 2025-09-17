import imageio
import os

filenames = [
    os.path.join("CSTR_NonLinear/Images", f)
    for f in os.listdir("CSTR_NonLinear/Images")
    if os.path.isfile(os.path.join("CSTR/Images", f))
]

with imageio.get_writer("CSTR_NonLinear/solution.gif", mode="I") as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        os.remove(filename)
