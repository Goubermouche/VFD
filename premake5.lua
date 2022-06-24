require('premake5-cuda')

workspace "FluidEngine"
    architecture "x64"

    configurations
    {
        "Debug",
        "Release"
    }

    startproject "FluidEngine"

outputdir = "{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"
-- Include directories relative to root folder (solution directory).
IncludeDir = {}
IncludeDir["GLFW"] = "FluidEngine/Vendor/GLFW/include"
IncludeDir["Glad"] = "FluidEngine/Vendor/Glad/include"
IncludeDir["ImGui"]= "FluidEngine/Vendor/imgui"
IncludeDir["glm"]  = "FluidEngine/Vendor/glm"
IncludeDir["entt"] = "FluidEngine/Vendor/entt/include"
IncludeDir["cereal"] = "FluidEngine/Vendor/cereal"

include "FluidEngine/Vendor/GLFW"
include "FluidEngine/Vendor/Glad"
include "FluidEngine/Vendor/imgui"

project "FluidEngine"
    location "FluidEngine"
    kind "ConsoleApp"
    language "C++"

    targetdir "bin/%{cfg.buildcfg}"

    pchheader "pch.h"
    pchsource "FluidEngine/src/pch.cpp"

    buildcustomizations "BuildCustomizations/CUDA 11.7"
    cudaPath "/usr/local/cuda" -- LINUX
    cudaMaxRegCount "32"

    cudaCompilerOptions 
    {
        "-arch=sm_52", 
        "-gencode=arch=compute_52,code=sm_52", 
        "-gencode=arch=compute_60,code=sm_60",
        "-gencode=arch=compute_61,code=sm_61", 
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_75,code=sm_75", 
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86", 
        "-gencode=arch=compute_86,code=compute_86",
         "-t0"
    }            

    files
    {
        "%{prj.name}/res/**.*",
        "%{prj.name}/src/**.h",
        "%{prj.name}/src/**.cpp",
        "%{prj.name}/src/**.txt",
        "%{prj.name}/src/**.shader",
        "%{prj.name}/Vendor/glm/**.hpp",
        "%{prj.name}/Vendor/glm/**.inl",
        "%{prj.name}/Vendor/cereal/**.hpp",
    }

    cudaFiles 
    {
        "**.cu",
        "**.cuh"
    }

    includedirs
    {
        "%{prj.name}/$(ProjectDir)src",
        "%{IncludeDir.GLFW}",
        "%{IncludeDir.Glad}",
        "%{IncludeDir.ImGui}",
        "%{IncludeDir.glm}",
        "%{IncludeDir.entt}",
        "%{IncludeDir.cereal}"
    }

    links
    {
        "GLFW",
        "Glad",
        "ImGui",
        "opengl32.lib",
        "cudart.lib"
    }

    if os.target() == "linux" then 
        linkoptions {"-L/usr/local/cuda/lib64 -lcudart"}
    end

    filter "system:windows"
        cppdialect "C++20"
        staticruntime "On"
        systemversion "latest"

        defines
        {
            "GLFW_INCLUDE_NONE"
        }

    filter "configurations:Debug"
        defines { "DEBUG" }
        symbols "On"
        staticruntime "off"
  
     filter "configurations:Release"
        defines { "NDEBUG" }
        optimize "Full"
        staticruntime "off"
        cudaFastMath "On"
