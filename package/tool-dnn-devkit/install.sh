
cd "$INSTALL_DIR"

cp $ORIGINAL_PACKAGE_DIR/src/* .

"$CK_CXX" "$CK_COMPILER_FLAG_CPP11" "$CK_OPT_SPEED" evaluate-object.cpp
mv a.out evaluate-object

exit 0
