import pyttsx3
import sys
from PyQt5.QtWidgets import QApplication
from app.ui_main import PoseMentor

def main():
    app = QApplication(sys.argv)
    window = PoseMentor() # PoseMentor class imported from app.ui_main
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()