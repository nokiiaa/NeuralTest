using NeuralLib;
using NeuralLib.Accel;
using NeuralLib.Layers;
using NeuralLib.Maths;
using NeuralLib.Misc;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Numerics;

namespace NeuralTest
{
    class Program
    {
        static void Usage()
        {
            Console.WriteLine($"Usage: new          <output config> <epochs> <learning rate> <batch size>");
            Console.WriteLine($"     | train        <input config> <output config> <epochs> <learning rate> <batch size>");
            Console.WriteLine($"     | test         <input config>");
            Console.WriteLine($"     | test_on_file <input config> <input bitmap>");
            Environment.Exit(1);
        }

        static void ReadMnistDatabase(bool training, bool testing, out List<(MArray, MArray)> examples, out List<(MArray, int)> tests)
        {
            byte[] trainImageData = training ? File.ReadAllBytes("train-images.idx3-ubyte") : null;
            byte[] trainLabelData = training ? File.ReadAllBytes("train-labels.idx1-ubyte") : null;
            byte[] testImageData = testing ? File.ReadAllBytes("t10k-images.idx3-ubyte") : null;
            byte[] testLabelData = testing ? File.ReadAllBytes("t10k-labels.idx1-ubyte") : null;

            examples = new List<(MArray, MArray)>();

            if (training)
            {
                for (int i = 0; i < 60000; i++)
                {
                    MArray arr = new MArray(1, 28 * 28);

                    for (int n = 0; n < 28 * 28; n++)
                        arr.Values[n] = trainImageData[16 + i * 28 * 28 + n] / 255f;

                    MArray expect = new MArray(1, 10);
                    expect[0, trainLabelData[8 + i]] = 1;

                    examples.Add((arr, expect));
                }

            }

            tests = new List<(MArray, int)>();

            if (testing)
            {
                for (int i = 0; i < 10000; i++)
                {
                    MArray arr = new MArray(1, 28 * 28);

                    for (int n = 0; n < 28 * 28; n++)
                        arr.Values[n] = testImageData[16 + i * 28 * 28 + n] / 255f;

                    tests.Add((arr, testLabelData[8 + i]));
                }
            }
        }

        static MArray ReadTestFile(string path)
        {
            Bitmap bmp = (Bitmap)Image.FromFile(path);

            MArray bmpInput = new MArray(28, 28);
            int i = 0;

            for (int y = 4; y < 24; y++)
            {
                for (int x = 4; x < 24; x++, i++)
                {
                    Color pixel = bmp.GetPixel(x - 4, y - 4);
                    bmpInput[x, y] = 1 - (pixel.R + pixel.G + pixel.B) / 768f;
                }
            }

            return bmpInput.View(1, 28 * 28);
        }

        static void Main(string[] args)
        {
            //args = new string[] { "test", "MLP.json" };
            args = new string[] { "new", "GPU_mlp_test.json", "10", "0.0005", "32" };
            //args = new string[] { "train", "MLP.json", "MLP_GPU.json", "10", "0.0005", "16" };

            List<(MArray, MArray)> examples = null;
            List<(MArray, int)> tests = null;

            Network network;

            int epochs = 0, batchSize = 0;
            float learningRate = 0;
            string outputFile = "";

            if (args.Length > 0)
            {
                if (args[0] == "new")
                {
                    if (args.Length < 5) Usage();

                    outputFile = args[1];

                    if (!int.TryParse(args[2], out epochs) ||
                        !float.TryParse(args[3].Replace('.', ','), out learningRate) ||
                        !int.TryParse(args[4], out batchSize))
                        Usage();

                    network = new Network
                    {
                        Layers = new List<Layer>()
                        {
                            new ConvLayer(28, 28, 1, 5, 5, 6, 0, Activations.Type.LeakyReLU, true, 1),
                            new PoolLayer(24, 24, 6, 2),
                            new ConvLayer(12, 12, 6, 5, 5, 16, 0, Activations.Type.LeakyReLU, true, 1),
                            new PoolLayer(8, 8, 16, 2),
                            new DenseLayer(4*4*16, 120, Activations.Type.LeakyReLU, true, 1),
                            new DenseLayer(120, 84, Activations.Type.LeakyReLU, true, 1),
                            new DenseLayer(84, 10, Activations.Type.LeakyReLU, true, 1),
                            new DenseLayer(10, 10, Activations.Type.LeakyReLU, true, 1),

                            /*new DenseLayer(28*28, 50, Activations.Type.LeakyReLU, true, 1),
                            new DenseLayer(50, 20, Activations.Type.LeakyReLU, true, 1),
                            new DenseLayer(20, 10, Activations.Type.LeakyReLU, true, 1),
                            new DenseLayer(10, 10, Activations.Type.LeakyReLU, true, 1)*/
                        }
                    };

                    ReadMnistDatabase(true, true, out examples, out tests);
                }
                else if (args[0] == "train")
                {
                    if (args.Length < 6)
                        Usage();

                    Console.WriteLine($"Importing network data from {args[1]}");

                    network = new Network
                    {
                        Layers = JsonConvert.DeserializeObject<List<Layer>>(File.ReadAllText(args[1]),
                        new JsonSerializerSettings() { Converters = new List<JsonConverter> { new NetworkConverter() } })
                    };

                    outputFile = args[2];

                    if (!int.TryParse(args[3], out epochs) ||
                        !float.TryParse(args[4].Replace('.', ','), out learningRate) ||
                        !int.TryParse(args[5], out batchSize))
                        Usage();

                    ReadMnistDatabase(true, true, out examples, out tests);
                }
                else if (args[0] == "test")
                {
                    Console.WriteLine($"Importing network data from {args[1]}");

                    network = new Network
                    {
                        Layers = JsonConvert.DeserializeObject<List<Layer>>(File.ReadAllText(args[1]),
                        new JsonSerializerSettings() { Converters = new List<JsonConverter> { new NetworkConverter() } })
                    };

                    ReadMnistDatabase(false, true, out examples, out tests);

                    goto testSkip;
                }
                else if (args[0] == "test_on_file")
                {
                    if (args.Length < 3)
                        Usage();

                    network = new Network
                    {
                        Layers = JsonConvert.DeserializeObject<List<Layer>>(File.ReadAllText(args[1]),
                        new JsonSerializerSettings() { Converters = new List<JsonConverter> { new NetworkConverter() } })
                    };

                    MArray output = network.Forward(ReadTestFile(args[2]));

                    Console.WriteLine($"Network sees this as a '{output.Values.ToList().IndexOf(output.Values.Max())}'");
                    Environment.Exit(0);
                }
                else
                {
                    Usage();
                    return;
                }

                Console.WriteLine($"Training for {epochs} epochs with batch size {batchSize} and learning rate {learningRate}");

                network.AdamSGD(epochs, learningRate, batchSize, examples.ToArray());
                
                Console.WriteLine("Training finished");

                string serialized = JsonConvert.SerializeObject(network.Layers);

                File.WriteAllText(outputFile, serialized);

                Console.WriteLine($"Network data serialized to {outputFile}");

            testSkip:
                int correct = 0;

                foreach ((MArray, int) test in tests)
                {
                    var output = network.Forward(test.Item1).Values.ToList();
                    int expected = test.Item2;
                    int result = output.IndexOf(output.Max());

                    if (expected == result)
                    {
                        Console.ForegroundColor = ConsoleColor.Green;
                        correct++;
                    }
                    else
                        Console.ForegroundColor = ConsoleColor.Red;

                    Console.WriteLine($"Predicted: {result} | True: {expected}");

                    Console.ForegroundColor = ConsoleColor.Gray;
                }

                Console.WriteLine($"Network accuracy = {(double)correct / tests.Count * 100}%");
                Console.ReadKey(true);
            }
            else
                Usage();
        }
    }
}