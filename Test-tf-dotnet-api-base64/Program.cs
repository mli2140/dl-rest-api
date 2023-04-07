using System.Drawing;
using ImageProcessor;
using Keras.Models;
using Keras.PreProcessing.Image;
using Numpy;

var model = Model.LoadModel("");

Console.Clear();

string imagestringNIO = ""; //base64 encoded image

byte[] base64_decoded = Convert.FromBase64String(imagestringNIO);


Size size = new Size(224, 224); //Preprocessing resize  this happens client side
Image imageBase64;

using (MemoryStream inStream = new MemoryStream(base64_decoded))
{
    using (MemoryStream outStream = new MemoryStream())
    {
        using (ImageFactory imageFactory = new ImageFactory(preserveExifData: true))
        {
            imageFactory.Load(inStream)
                        .Resize(size)
                        .Save(outStream);
        }
        imageBase64 = Image.FromStream(outStream);
    }
}

static byte[] ImageToByte2(Image img)
{
    using (var stream = new MemoryStream())
    {
        img.Save(stream, System.Drawing.Imaging.ImageFormat.Png);
        return stream.ToArray();
    }
}

byte[] test = ImageToByte2(imageBase64);


string image_path = "";

var image = ImageUtil.LoadImg(
                               path: image_path,
                               target_size: new Keras.Shape(224, 224),
                               color_mode: "rgb");

NDarray ImageArray = ImageUtil.ImageToArray(image);

//Create NDarray Prototype of Image Shape (224,224,3)
NDarray three = new(new float[3]);
NDarray[] firstLayer224 = Enumerable.Repeat(three, 224).ToArray();
NDarray stacked = np.stack(firstLayer224);
NDarray[] secondLayer224 = Enumerable.Repeat(stacked, 224).ToArray();
NDarray emptyImageArrayShaped = np.stack(secondLayer224);

//Preparing Image Array for prediction
ImageArray = np.expand_dims(emptyImageArrayShaped, axis: 0);
NDarray imagePredict = np.vstack(new NDarray[] { ImageArray });

NDarray val = model.Predict(imagePredict);

//Generic Function to read Prediction Data From prediction
float[] testValues = val.GetData<float>();

if (testValues[0] == 0.0)
{
    Console.WriteLine("Prediction: IO");
}
else
{
    Console.WriteLine("Prediction: NIO");
}
Console.WriteLine($"Confidence: {val[0, 0]}");
