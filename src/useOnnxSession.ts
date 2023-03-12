import { useState, useRef, useEffect } from "react";
import { InferenceSession } from "onnxruntime-web";

export type RunProps = {
  feeds: InferenceSession.OnnxValueMapType;
  options?: InferenceSession.RunOptions;
};

/**
 * @param model Path to .onnx model file
 * @param options https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html
 * @returns A function that runs the inference session using the provided input
 */
export const useOnnxSession = (
  model: string,
  options?: InferenceSession.SessionOptions
) => {
  const [session, setSession] = useState<InferenceSession>();
  const makingSession = useRef(false);

  const makeSession = async () => {
    makingSession.current = true;
    const sess = await InferenceSession.create(model, options);
    setSession(sess);
  };

  useEffect(() => {
    if (!session && !makingSession.current) {
      makeSession();
    }
  }, []);

  const requestInference = async ({ feeds, options }: RunProps) => {
    // TODO: Make sure this can't be called too soon
    return await session!.run(feeds, options);
  };

  return requestInference;
};
