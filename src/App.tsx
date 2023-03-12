import { KeyboardEventHandler, useEffect, useRef, useState } from "react";
import { InferenceSession, Tensor } from "onnxruntime-web";
import { Character } from "./components/Character";
import { runModel, isAlphaNum } from "./utils";
import { addresses } from "./components/addresses";
import styled from "styled-components";
import { useOnnxSession } from "./useOnnxSession";
import { useInterval } from "./useInterval";

const AppWrap = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  padding: 20px;
`;

const Cursor = styled.div`
  width: 3px;
  height: 100px;
  background-color: blue;
  display: inline-block;
`;

type LetterForm = {
  char: string[];
  currAddr?: number[];
  image: InferenceSession.OnnxValueMapType;
};

function App() {
  const appRef = useRef<HTMLDivElement>(null);
  const [chars, setChars] = useState<LetterForm[]>([]);
  const [heldKeys, setHeldKeys] = useState<Set<string>>(new Set([]));
  const destAddress = useRef<number[]>();
  const didArrive = useRef(false);
  const [cursorPos, setCursorPos] = useState(0);

  // focus the typing area immediately
  useEffect(() => appRef.current?.focus());

  useEffect(() => {
    console.log(cursorPos);
  }, [cursorPos]);

  // create ONNX session and get the inference function
  const requestInference = useOnnxSession("./emnist_vgan.onnx", {
    executionProviders: ["webgl"],
    graphOptimizationLevel: "all",
  });

  // setup animation interval for when keys are held
  useInterval(handleHeldKeys, heldKeys.size ? 10 : null);

  const handleKeyDown: KeyboardEventHandler = async (e) => {
    console.log(e);

    // reset didArrive whenever a new key is pressed
    if (![...heldKeys].includes(e.key)) {
      didArrive.current = false;
    }

    // update heldKeys
    if (isAlphaNum(e.key)) {
      const newHeldKeys = new Set([...heldKeys]);
      newHeldKeys.add(e.key);
      setHeldKeys(newHeldKeys);
    }

    // handle non-letter keys
    if (e.key == "Backspace") {
      doBackspace(chars, setChars);
    } else if (e.key == " ") {
      doSpace(chars, setChars);
    } else if (isAlphaNum(e.key) && !heldKeys.size) {
      doLetter(e.key, chars, setChars);
    } else if (e.key == "ArrowRight") {
      setCursorPos(Math.min(cursorPos + 1, chars.length));
    } else if (e.key == "ArrowLeft") {
      setCursorPos(Math.max(cursorPos - 1, 0));
    }
    // TODO: LINEBREAK
    // TODO: HANDLE SHIFT+KEY
  };

  const handleKeyUp: KeyboardEventHandler = (e) => {
    const newHeldKeys = new Set([...heldKeys]);
    newHeldKeys.delete(e.key);
    setHeldKeys(newHeldKeys);
    didArrive.current = false;
  };

  async function handleHeldKeys() {
    const heldKeyArr = [...heldKeys];
    const currAddr = chars[chars.length - 1].currAddr;

    // check if we have arrived at destAddress
    if (currAddr?.toString() == destAddress?.current?.toString()) {
      console.log("arrived");
      didArrive.current = true;
    }

    if (heldKeyArr.length == 1) {
      // destAddress corresponds to a single letter
      destAddress.current = addresses[heldKeyArr[0]];
    } else {
      // gather addresses to average
      const heldAddrs = heldKeyArr.map((key) => addresses[key]);

      // generate mean address
      let result = [];
      for (let i = 0; i < heldAddrs[0].length; i++) {
        let num = 0;
        for (let j = 0; j < heldAddrs.length; j++) {
          num += heldAddrs[j][i];
        }
        result.push(num / heldAddrs.length);
      }

      // set mean as destination
      destAddress.current = result;
    }

    if (currAddr) {
      if (!didArrive.current) {
        // lerp toward the destination address
        const incr = 0.1;
        const newAddr = [];
        for (const [i, curr] of currAddr.entries()) {
          const dest = destAddress.current[i];
          const diff = dest - curr;

          if (Math.abs(diff) <= incr) {
            // if increment would go past dest, go to dest
            newAddr.push(dest);
          } else {
            // otherwise, increment toward dest
            newAddr.push(curr + Math.sign(diff) * incr);
          }
        }

        const newLastChar: LetterForm = {
          char: heldKeyArr,
          currAddr: newAddr,
          image: await runModel(newAddr, requestInference),
        };

        setChars([...chars.slice(0, -1), newLastChar]);
      } else {
        // drunkard's walk around current address
        console.log("explore");
        const newAddr = [];
        for (const num of currAddr) {
          const incr = Math.random() < 0.5 ? -0.05 : 0.05;
          newAddr.push(num + incr);
        }

        const newLastChar: LetterForm = {
          char: heldKeyArr,
          currAddr: newAddr,
          image: await runModel(newAddr, requestInference),
        };

        setChars([...chars.slice(0, -1), newLastChar]);
      }
    }
  }

  const doBackspace = (
    chars: LetterForm[],
    setChars: (a: LetterForm[]) => void
  ) => {
    setChars([...chars.slice(0, cursorPos - 1), ...chars.slice(cursorPos)]);
    setCursorPos(Math.max(cursorPos - 1, 0));
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
    setChars: (a: LetterForm[]) => void
  ) => {
    console.log("DO LETTER");

    setChars([
      ...chars,
      {
        char: [key],
        currAddr: addresses[key],
        image: await runModel(addresses[key], requestInference),
      },
    ]);

    // advance the cursor
    setCursorPos(Math.min(cursorPos + 1, chars.length));
  };

  return (
    <div className="App">
      <AppWrap
        ref={appRef}
        tabIndex={0}
        onKeyDown={handleKeyDown}
        onKeyUp={handleKeyUp}
      >
        {!chars.length || cursorPos == 0 ? <Cursor /> : null}
        {chars.map((c, i) => (
          <>
            <Character key={i} chars={c.char} nnOutput={c.image} />
            {i == cursorPos - 1 && cursorPos > 0 ? <Cursor /> : null}
          </>
        ))}
      </AppWrap>
    </div>
  );
}

export default App;
