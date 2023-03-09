import {
  KeyboardEventHandler,
  MouseEventHandler,
  useEffect,
  useRef,
  useState,
} from "react";
import { Character } from "./components/Character";
import * as ort from "onnxruntime-web";
import { imgFromArray, isAlphaNum } from "./utils";
import { useUpdateEffect } from "./utils";
import { addresses } from "./components/addresses";
import styled from "styled-components";

const AppWrap = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
`;

const StyledLetterForm = styled.span`
  width: 28px;
  height: 28px;
  border: 1px solid black;
  vertical-align: top;
  display: inline-block;
`;

const useOnnxSession = ({ modelPath }: { modelPath: string }) => {
  // global state:
  // is model running
  // [ 'a' 'b' 'c']
  // [ [....], [...], [....]]

  const [session, setSession] = useState<ort.InferenceSession>();
  const makingSession = useRef(false);
  const jobs = useRef<number[][]>([]);

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

  const addJob = (address: number[], callback: (img: any) => void) => {
    jobs.current.push(address);
  };

  return session;
};

type LetterForm = {
  char: string[];
  image: number[] | "pending";
};

function App() {
  const keyHeld = useRef(false);
  const appRef = useRef<HTMLDivElement>(null);
  const [chars, setChars] = useState<LetterForm[]>([]);

  // TODO: Memoize this
  const handleKey: KeyboardEventHandler = (e) => {
    if (e.key == "Backspace") {
      setChars(chars.slice(0, -1));
    } else if (!keyHeld.current) {
      // handle adding new character, disabled if still holding previous key
      keyHeld.current = true;
      if (isAlphaNum(e.key) || e.key == " ") {
        setChars([...chars, { char: [e.key], image: "pending" }]);
      }
    } else if (
      // hold-to-hybridize, where we add a letter
      isAlphaNum(e.key) &&
      isAlphaNum(chars[chars.length - 1].char[0]) &&
      !chars[chars.length - 1].char.includes(e.key)
    ) {
      const newLastChar: LetterForm = {
        char: [...chars[chars.length - 1].char, e.key],
        image: "pending",
      };
      setChars([...chars.slice(0, -1), newLastChar]);
    }
  };

  useEffect(() => {
    appRef.current?.focus();
  });

  useEffect(() => {
    console.log(chars);
  }, [chars]);

  return (
    <div className="App">
      <AppWrap
        ref={appRef}
        tabIndex={0}
        onKeyDown={handleKey}
        onKeyUp={() => (keyHeld.current = false)}
      >
        {chars.map((c, i) => (
          <StyledLetterForm key={i}>
            {c.char.map((l, j) => (
              <span key={j}>{l}</span>
            ))}
          </StyledLetterForm>
        ))}
      </AppWrap>
    </div>
  );
}

export default App;
