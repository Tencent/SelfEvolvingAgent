import { createContext } from "react";

type CKWSType = {
  CKWSConnection: WebSocket | null;
};

interface CKWSContextType {
  CKWSConnection: CKWSType;
  setCKWSConnection: (
    CKWSConnection: CKWSType | ((prevStatus: CKWSType) => CKWSType)
  ) => void;
}

export const CKWSContext = createContext<CKWSContextType | undefined>(
  undefined
);
export type { CKWSType };
export type { CKWSContextType };
