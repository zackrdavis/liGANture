import {
  KeyboardEventHandler,
  MouseEventHandler,
  useEffect,
  useRef,
  useState,
} from "react";
import { Character } from "./components/Character";
import * as ort from "onnxruntime-web";
import { isAlphaNum } from "./utils";
import { addresses } from "./components/addresses";
import styled from "styled-components";
import * as lerp_array from "lerp-array";
import { useOnnxSession } from "./useOnnxSession";
import { Tensor } from "onnxruntime-web";

const AppWrap = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  padding: 20px;
`;

type LetterForm = {
  char: string[];
  image: ort.InferenceSession.OnnxValueMapType;
};

function App() {
  const [keyHeld, setKeyHeld] = useState(false);
  const appRef = useRef<HTMLDivElement>(null);
  const [chars, setChars] = useState<LetterForm[]>([]);
  const lastKey = useRef<string>();

  const { requestInference } = useOnnxSession("./emnist_vgan.onnx");

  const handleKeyDown: KeyboardEventHandler = async (e) => {
    const isRepeatedKey = lastKey.current == e.key;
    lastKey.current = e.key;

    if (e.key == "Backspace") {
      doBackspace(chars, setChars);
    } else if (e.key == " ") {
      doSpace(chars, setChars);
    } else if (isAlphaNum(e.key)) {
      if (!keyHeld) {
        doLetter(e.key, chars, setChars, requestInference);
      } else {
        if (isRepeatedKey) {
          // animate lerp to areas near held letter
          await doExploreNear(e.key, chars, setChars, requestInference);
        } else {
          // animate lerp between the last held letter and the new letter
          await doHybridLetter(e.key, chars, setChars, requestInference);
        }
      }
      setKeyHeld(true);
    }
  };

  useEffect(() => {
    appRef.current?.focus();
  });

  const doExploreNear = async (
    key: string,
    chars: LetterForm[],
    setChars: (a: LetterForm[]) => void,
    requestInference: (
      address: number[]
    ) => Promise<ort.InferenceSession.OnnxValueMapType>
  ) => {
    console.log("holding same");
  };

  const doBackspace = (
    chars: LetterForm[],
    setChars: (a: LetterForm[]) => void
  ) => {
    setChars(chars.slice(0, -1));
  };

  const doSpace = (
    chars: LetterForm[],
    setChars: (a: LetterForm[]) => void
  ) => {
    setChars([
      ...chars,
      {
        char: [" "],
        image: { img: new Tensor("float32", Array(784).fill(-1), [1, 784]) },
      },
    ]);
  };

  const doLetter = async (
    key: string,
    chars: LetterForm[],
    setChars: (a: LetterForm[]) => void,
    requestInference: (
      address: number[]
    ) => Promise<ort.InferenceSession.OnnxValueMapType>
  ) => {
    setChars([
      ...chars,
      {
        char: [key],
        image: await requestInference(addresses[key]),
      },
    ]);
  };

  const doHybridLetter = async (
    key: string,
    chars: LetterForm[],
    setChars: (a: LetterForm[]) => void,
    requestInference: (
      address: number[]
    ) => Promise<ort.InferenceSession.OnnxValueMapType>
  ) => {
    console.log("holding multiple");

    const addressesToCombine = [
      [...addresses[chars[chars.length - 1].char[0]]],
      [...addresses[key]],
    ];

    for (const step of [
      0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    ]) {
      const hybridAddress = lerp_array(...addressesToCombine, step);

      const newLastChar: LetterForm = {
        char: [...chars[chars.length - 1].char, key],
        image: await requestInference(hybridAddress),
      };

      setChars([...chars.slice(0, -1), newLastChar]);
    }
  };

  return (
    <div className="App">
      <AppWrap
        ref={appRef}
        tabIndex={0}
        onKeyDown={handleKeyDown}
        onKeyUp={() => setKeyHeld(false)}
      >
        {chars.map((c, i) => (
          <Character key={i} chars={c.char} nnOutput={c.image} />
        ))}
      </AppWrap>
    </div>
  );
}

export default App;
