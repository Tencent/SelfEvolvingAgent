import { createContext } from "react";

type MyMessageStateType = {
  showEditBox: boolean;
  editText: any;
  currentMessageId: string | null;
};

type MyMessageSegmentationType = {
  group: string;
  content: string;
  pos: number;
};

type MyMessageType = {
  message: MyMessageSegmentationType[];
  messageState: MyMessageStateType;
  sender: string;
};

type CKStatusType = {
  current_model: string;
  activated_websites: string[];
  activated_functions_personalized_model: boolean;
  activated_functions_self_improving: boolean;
  activated_functions_online_annotation: boolean;
  uploaded_files: string[];
  messages: MyMessageType[];
  session_id: string;
  history_id: string;
  username: string;
  currentMode: string;
};

interface StatusContextType {
  CKStatus: CKStatusType;
  setCKStatus: (
    CKStatus: CKStatusType | ((prevStatus: CKStatusType) => CKStatusType)
  ) => void;
}

export const StatusContext = createContext<StatusContextType | undefined>(
  undefined
);
export type { MyMessageStateType };
export type { MyMessageSegmentationType };
export type { MyMessageType };
export type { CKStatusType };
