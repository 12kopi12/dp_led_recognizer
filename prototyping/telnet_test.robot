*** Settings ***
Library    SerialLibrary
Library    TelnetCommands

*** Test Cases ***

TEST_ERROR_COMMAND
    ${RESPONSE} =    Send Command    er?
    Should Contain    ${RESPONSE}    ERR

TEST_SERVICE_MODE
    Log    Service mode dym=100 -> DS=00
    ${RESPONSE} =    Send Command    DYM=100
    Should Match    "${RESPONSE}"    "DS=00"