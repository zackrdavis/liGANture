import { useState, useRef, useEffect } from "react";
import * as ort from "onnxruntime-web";

const sleep = (ms: number) => {
  return new Promise((resolve) => setTimeout(resolve, ms));
};

export const useOnnxSession = (modelPath: string) => {
  const [session, setSession] = useState<ort.InferenceSession>();
  const makingSession = useRef(false);

  // create this session just once for the whole app
  const makeSession = async () => {
    makingSession.current = true;

    ort.InferenceSession.create(modelPath, {
      executionProviders: ["webgl"],
      graphOptimizationLevel: "all",
    }).then((sess) => setSession(sess));
  };

  useEffect(() => {
    if (!session && !makingSession.current) {
      makeSession();
    }
  }, []);

  const requestInference = async (address: number[]) => {
    await sleep(20);
    // shape the address to work with the model
    const tensor = new ort.Tensor("float32", address, [1, 100]);
    // return a promise
    return await session!.run({ z: tensor });
  };

  return { requestInference };
};
